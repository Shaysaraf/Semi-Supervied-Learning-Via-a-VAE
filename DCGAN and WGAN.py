import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix Intel OpenMP runtime conflict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
Z_DIM = 100
BATCH_SIZE = 64
N_CRITIC = 10
LAMBDA_GP = 10
LR = 5e-5
BETA1 = 0.0
BETA2 = 0.9
MAX_EPOCHS = 25
IMAGE_SIZE = 28

# Early stopping parameters
LOSS_WINDOW = 5
MIN_EPOCHS = 10
EARLY_STOP_THRESHOLD = 1e-4

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = datasets.FashionMNIST(root='~/.pytorch/F_MNIST_data/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Critic with InstanceNorm2d
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.model(x)

# Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=DEVICE)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Initialize models
G = Generator(Z_DIM).to(DEVICE)
D = Critic().to(DEVICE)
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

# Training loop
g_losses, d_losses, prev_d_losses = [], [], []

print("Training WGAN-GP...\n")
total_start_time = time.time()

for epoch in range(MAX_EPOCHS):
    epoch_start_time = time.time()

    for i, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # Train Critic
        for _ in range(N_CRITIC):
            z = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_imgs = G(z).detach()

            real_validity = D(real_imgs)
            fake_validity = D(fake_imgs)
            gp = compute_gradient_penalty(D, real_imgs, fake_imgs)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gp

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

        # Train Generator
        z = torch.randn(batch_size, Z_DIM, device=DEVICE)
        gen_imgs = G(z)
        g_loss = -torch.mean(D(gen_imgs))

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

    # Progress
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | "
          f"Time: {epoch_time:.1f}s")

    # Early stopping check
    prev_d_losses.append(d_loss.item())
    if len(prev_d_losses) > LOSS_WINDOW:
        prev_d_losses.pop(0)
        delta = np.abs(np.mean(np.diff(prev_d_losses)))
        if delta < EARLY_STOP_THRESHOLD and epoch + 1 >= MIN_EPOCHS:
            print(f"\nEarly stopping at epoch {epoch+1}. ΔLoss: {delta:.6f}")
            break

# Final stats
total_time = time.time() - total_start_time
print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} minutes)")

# Plot loss
plt.plot(d_losses, label="Critic Loss")
plt.plot(g_losses, label="Generator Loss")
plt.title("WGAN-GP Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Generate and show images
def show_generated(generator, n=5):
    generator.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    with torch.no_grad():
        imgs = generator(z).cpu()
    imgs = imgs * 0.5 + 0.5  # Denormalize
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        axs[i].imshow(imgs[i].squeeze(), cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Generated Images (WGAN-GP)")
    plt.show()

def show_real_images():
    real_imgs, _ = next(iter(train_loader))
    real_imgs = real_imgs[:5] * 0.5 + 0.5
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(real_imgs[i].squeeze(), cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Real Fashion MNIST Images")
    plt.show()

# Display final results
show_real_images()
show_generated(G, n=5)
