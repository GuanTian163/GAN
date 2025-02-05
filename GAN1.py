import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad
import torchvision.utils as vutils

# Preprocessing Function
def preprocess_images(input_folder, output_folder, image_size=(64, 64)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
    ])
    os.makedirs(output_folder, exist_ok=True)
    for idx, category in enumerate(sorted(os.listdir(input_folder))):
        category_path = os.path.join(input_folder, category)
        if os.path.isdir(category_path):
            output_category_path = os.path.join(output_folder, category)
            os.makedirs(output_category_path, exist_ok=True)
            for filename in os.listdir(category_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(category_path, filename)
                    image = Image.open(image_path).convert('RGB')
                    image = transform(image)
                    image.save(os.path.join(output_category_path, filename))
                    print(f"Processed {filename} in {category}")

# Training Code Wrapped in __main__ Guard
if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    nz = 100       # Latent vector size
    ngf = 64       # Feature maps in generator
    ndf = 64       # Feature maps in discriminator
    nc = 3         # Number of color channels
    batch_size = 64
    num_epochs = 500
    lr = 0.0001      # Lower learning rate for stability
    beta1 = 0.1
    beta2 = 0.9
    lambda_gp = 5   # Adjusted gradient penalty
    n_critic = 5     # Discriminator updates per generator update
    embedding_dim = 50  # Embedding dimension for conditional input

    # Preprocess Images
    input_folder = 'dataset'
    output_folder = 'preprocessed_dataset'
    preprocess_images(input_folder, output_folder, image_size=(64, 64))

    # Data Loaders
    data_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5], [0.5])
    ])
    
    dataset = datasets.ImageFolder(root=output_folder, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Determine number of classes dynamically
    num_classes = len(dataset.classes)
    print(f"Classes: {dataset.classes}")
    print(f"Number of classes: {num_classes}")

    # Define Weight Initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    # Generator and Discriminator Classes
    class Generator(nn.Module):
        def __init__(self, nz, ngf, nc, num_classes, embedding_dim):
            super(Generator, self).__init__()
            self.nz = nz
            self.label_emb = nn.Embedding(num_classes, embedding_dim)
            self.init_size = 64 // 4
            self.l1 = nn.Sequential(nn.Linear(nz + embedding_dim, ngf * 8 * self.init_size ** 2))

            self.main = nn.Sequential(
                nn.BatchNorm2d(ngf * 8),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 8, ngf * 4, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 4, ngf * 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.Conv2d(ngf * 2, nc, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise, labels):
            labels = self.label_emb(labels)
            gen_input = torch.cat((noise, labels), -1)
            out = self.l1(gen_input)
            out = out.view(out.size(0), ngf * 8, self.init_size, self.init_size)
            img = self.main(out)
            return img

    class Discriminator(nn.Module):
        def __init__(self, nc, ndf, num_classes, embedding_dim):
            super(Discriminator, self).__init__()
            self.label_embedding = nn.Embedding(num_classes, embedding_dim)

            self.main = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(nc + embedding_dim, ndf, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, 1, 4, 1, 0),
            )

        def forward(self, img, labels):
            labels = self.label_embedding(labels)
            labels = labels.unsqueeze(2).unsqueeze(3)
            labels = labels.expand(labels.size(0), labels.size(1), img.size(2), img.size(3))
            d_in = torch.cat((img, labels), 1)
            validity = self.main(d_in)
            return validity.view(-1)

    # Model Initialization
    netG = Generator(nz, ngf, nc, num_classes, embedding_dim).to(device)
    netD = Discriminator(nc, ndf, num_classes, embedding_dim).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # Initialize lists for storing losses
    G_losses = []
    D_losses = []

    # Training Loop
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            real_images, labels = real_images.to(device), labels.to(device)
            b_size = real_images.size(0)

            # Train Discriminator
            netD.zero_grad()
            real_validity = netD(real_images, labels)
            D_real = real_validity.mean()

            noise = torch.randn(b_size, nz, device=device)
            gen_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_images = netG(noise, gen_labels)
            fake_validity = netD(fake_images.detach(), gen_labels)
            D_fake = fake_validity.mean()

            # Gradient penalty
            alpha = torch.rand(b_size, 1, 1, 1, device=device)
            interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
            interpolates_validity = netD(interpolates, labels)
            gradients = grad(
                outputs=interpolates_validity,
                inputs=interpolates,
                grad_outputs=torch.ones(interpolates_validity.size(), device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            errD = D_fake - D_real + gradient_penalty
            errD.backward()
            optimizerD.step()

            # Train the generator every n_critic steps
            if i % n_critic == 0:
                netG.zero_grad()
                # Regenerate fake images for generator step to ensure graph is fresh
                noise = torch.randn(b_size, nz, device=device)
                gen_labels = torch.randint(0, num_classes, (b_size,), device=device)
                fake_images = netG(noise, gen_labels)
                fake_validity = netD(fake_images, gen_labels)
                errG = -fake_validity.mean()
                errG.backward()
                optimizerG.step()

                G_losses.append(errG.item())
                D_losses.append(errD.item())

            if i % 200 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}')

        # Generate sample images every 10 epochs
        if epoch % 10 == 0:
            netG.eval()
            with torch.no_grad():
                for idx, class_name in enumerate(dataset.classes):
                    noise = torch.randn(16, nz, device=device)
                    labels = torch.full((16,), idx, dtype=torch.long, device=device)
                    generated_images = netG(noise, labels).cpu()
                    grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
                    np_grid = grid.permute(1, 2, 0).numpy()
                    plt.imsave(f'generated_{class_name}_epoch{epoch}.png', np_grid, format="png")
            netG.train()

    # Save loss plot
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")

    # Save Model
    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')
