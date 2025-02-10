# GAN for Image Generation
This work, titled “Generative Adversarial Networks for Sustainable Urban and Ecological Visualization in Digital Media Art,” is published in The Visual Computer


## Overview
This repository contains a Generative Adversarial Network (GAN) for image generation. The model consists of a conditional generator and a discriminator that leverage **spectral normalization** and **gradient penalty** for enhanced training stability. It is implemented using **PyTorch** and trained on **custom datasets** processed through torchvision transformations.

## Features
- **Conditional GAN:** The model generates images conditioned on class labels.
- **Improved Training Stability:** Utilizes **gradient penalty** (λ = 5) and **spectral normalization**.
- **Automatic Dataset Preprocessing:** Handles image resizing, cropping, and normalization.
- **Dynamic Class Support:** Adapts to any dataset with labeled subfolders.
- **Loss Tracking & Image Logging:** Saves loss plots and generated images per class every 10 epochs.

---

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/GuanTian163/GAN.git
cd GAN
```

### 2. Install Dependencies
Ensure you have **Python 3.7+** installed. Then install the required libraries:
```bash
pip install torch torchvision matplotlib numpy pillow
```

---

## Dataset Preparation
The model expects images organized in subfolders under a `dataset/` directory.
```plaintext
dataset/
├── class_1/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── ...
├── class_2/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── ...
```
After running the script, images are preprocessed and stored in `preprocessed_dataset/`.

---

## Running the Model
### 1. Preprocess Images
```bash
python train.py
```
This script:
- Resizes images to **64x64**
- Normalizes pixel values
- Creates a `preprocessed_dataset/` directory

### 2. Training the GAN
The model is trained using **500 epochs** with a batch size of **64**.
```bash
python train.py
```
The training process includes:
- Dynamic class detection
- Conditional GAN training
- Loss visualization & periodic sample generation

---

## Model Details
### Generator
- **Input:** Random noise vector (`nz=100`) + class embedding (`embedding_dim=50`)
- **Architecture:**
  - Linear layer expansion (Xavier initialization)
  - 4 upsampling convolutional layers
  - Batch normalization and ReLU activations
  - **Tanh output layer** for image generation

### Discriminator
- **Input:** Real/fake image + class embedding
- **Architecture:**
  - Spectral normalization on convolutional layers
  - Leaky ReLU activation
  - 4 convolutional layers
  - **Final layer outputs validity score**

---

## Training Strategy
- **Hyperparameters:**
  - Learning Rate: `0.0001`
  - Beta1: `0.1`, Beta2: `0.9`
  - Gradient Penalty: `λ=5`
- **Losses Monitored:**
  - Generator Loss (`-D(G(z))`)
  - Discriminator Loss (`D(real) - D(fake) + gradient_penalty`)

### Sample Generation
- Every **10 epochs**, the model saves generated images for each class.
- Output images are stored as `generated_classname_epochX.png`.

---

## Model Saving
- After training, the generator and discriminator weights are stored as:
```plaintext
./generator.pth
./discriminator.pth
```
- Loss curves are saved as `loss_plot.png`.


## Citation Format
If you use this code, please cite:
```plaintext
Guan Tian, "Generative Adversarial Networks for Sustainable Urban and Ecological Visualization in Digital Media Art," The Visual Computer, 2025.


## Contact
For any questions, please contact [Guan Tian](15342348802@163.com).


