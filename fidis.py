import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models.inception import inception_v3
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import linalg  # For matrix operations like sqrtm

# Custom Dataset for loading images from multiple folders (class names from subdirectories)
class CustomImageDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        self.folder_paths = folder_paths
        self.transform = transform
        self.image_paths = []

        for folder_path in folder_paths:
            # Ensure the folder exists
            if os.path.exists(folder_path):
                # Add the images from the folder to the list
                self.image_paths += [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
            else:
                print(f"Warning: Folder {folder_path} does not exist!")

        # Debugging: Print paths of the images being loaded
        print(f"Found {len(self.image_paths)} images in the following folders: {folder_paths}")
        for img_path in self.image_paths[:5]:  # Print the first few image paths for verification
            print(f"Image: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Preprocessing function
def preprocess_images(input_folders, image_size=(75, 75)):  # Increase the size to 75x75 for InceptionV3
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    dataset = CustomImageDataset(input_folders, transform=transform)
    return dataset

# FID Calculation function
def calculate_fid(real_dataloader, generated_dataloader):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    def get_features(dataloader):
        features = []
        for data in dataloader:
            data = data.to(device)
            with torch.no_grad():
                feature = model(data)
            features.append(feature.cpu().numpy())
        if len(features) == 0:
            print("Warning: No features extracted. Check your dataloader.")
            return np.array([])  # Return an empty array if no features are collected
        return np.concatenate(features, axis=0)

    real_features = get_features(real_dataloader)
    generated_features = get_features(generated_dataloader)

    # Ensure that both features arrays are not empty before calculating FID
    if real_features.size == 0 or generated_features.size == 0:
        return None  # Return None if features are empty, indicating FID cannot be calculated

    # Calculate FID
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_generated = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_generated
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_generated), disp=False)

    # Check if the result has complex values and only keep the real part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2.0 * covmean)
    return fid

# IS Calculation function
def calculate_inception_score(generated_dataloader):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    def get_probabilities(dataloader):
        probabilities = []
        for data in dataloader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            probabilities.append(F.softmax(output, dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    probabilities = get_probabilities(generated_dataloader)

    # Compute Inception Score
    kl_div = np.sum(probabilities * (np.log(probabilities) - np.log(np.mean(probabilities, axis=0))), axis=1)
    inception_score = np.exp(np.mean(kl_div))

    return inception_score

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(999)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Folder paths for real and generated images
    real_images_folders = [
        "E:/Generative Adversarial Networks/Images/preprocessed_dataset/natural_habitats",
        "E:/Generative Adversarial Networks/Images/preprocessed_dataset/urban_greenery",
        "E:/Generative Adversarial Networks/Images/preprocessed_dataset/eco_friendly_practices"
    ]
    generated_images_folder = "E:/Generative Adversarial Networks/Images/generated Images"

    # Check if real image folders are empty
    for folder in real_images_folders:
        if not os.listdir(folder):
            print(f"Warning: Real images folder {folder} is empty!")

    # Check if generated image folder is empty
    if not os.listdir(generated_images_folder):
        print(f"Warning: Generated images folder {generated_images_folder} is empty!")

    # Preprocess the images from real and generated datasets
    real_images = preprocess_images(real_images_folders, image_size=(75, 75))  # Increased image size
    generated_images = preprocess_images([generated_images_folder], image_size=(75, 75))  # Increased image size

    # Create DataLoader for batching
    real_dataloader = DataLoader(real_images, batch_size=64, shuffle=False)
    generated_dataloader = DataLoader(generated_images, batch_size=64, shuffle=False)

    # Check if the DataLoader has images
    print(f"Number of images in real data: {len(real_images)}")
    print(f"Number of images in generated data: {len(generated_images)}")

    # Extract features from one batch to check if feature extraction works
    print("Extracting features from one batch of real images...")
    real_batch = next(iter(real_dataloader))
    print(f"Batch size: {real_batch.size()}")  # Check the batch size

    print("Extracting features from one batch of generated images...")
    generated_batch = next(iter(generated_dataloader))
    print(f"Batch size: {generated_batch.size()}")  # Check the batch size

    # FID and Inception Score Calculation
    fid_score = None  # Initialize FID as None to handle errors gracefully

    try:
        fid_score = calculate_fid(real_dataloader, generated_dataloader)
        if fid_score is not None:
            print(f"FID Score: {fid_score}")
        else:
            print("FID could not be calculated due to empty feature sets.")
    except ValueError as e:
        print(f"Error calculating FID: {e}")

    try:
        inception_score = calculate_inception_score(generated_dataloader)
        print(f"Inception Score: {inception_score}")
    except Exception as e:
        print(f"Error calculating Inception Score: {e}")

    # Plot and Save Scores as PNG
    if fid_score is not None:
        fig, ax = plt.subplots()
        ax.bar(['FID', 'Inception Score'], [fid_score, inception_score])
        ax.set_title("FID and Inception Score")
        plt.savefig("score_plot.png")  # Save the plot as PNG
        print("Scores plot saved as score_plot.png")
    else:
        print("FID Score not available, plot not generated.")
