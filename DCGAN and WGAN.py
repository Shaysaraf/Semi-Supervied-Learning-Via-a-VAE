import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure CPU compatibility and avoid multiprocessing slowdown
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Device
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
Z_DIM = 100
BATCH_SIZE = 64
N_CRITIC = 5
LAMBDA_GP = 10
LR = 1e-4
BETA1 = 0.0
BETA2 = 0.9
MAX_EPOCHS = 40

# DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.ReLU(inplace=False),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Critic
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.model(x)

# Weight Init
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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
G.apply(weights_init)
D.apply(weights_init)

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

g_losses, d_losses = [], []

print("Starting WGAN-GP Training...\n")
start_time = time.time()

for epoch in range(MAX_EPOCHS):
    epoch_start = time.time()
    data_iter = iter(train_loader)
    i = 0

    while i < len(train_loader):
        for _ in range(N_CRITIC):
            try:
                real_imgs, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                real_imgs, _ = next(data_iter)

            i += 1
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            z = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_imgs = G(z).detach()

            real_validity = D(real_imgs)
            fake_validity = D(fake_imgs)
            gp = compute_gradient_penalty(D, real_imgs, fake_imgs)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gp

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

        # Generator update
        z = torch.randn(batch_size, Z_DIM, device=DEVICE)
        gen_imgs = G(z)
        g_loss = -torch.mean(D(gen_imgs))

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    epoch_duration = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Time: {epoch_duration:.1f}s")

# Loss plot
plt.plot(d_losses, label='Critic Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('WGAN-GP Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Show samples
def show_generated(generator, n=5):
    generator.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    with torch.no_grad():
        imgs = generator(z).cpu()
    imgs = imgs * 0.5 + 0.5
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        axs[i].imshow(imgs[i].squeeze(), cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Generated Images (WGAN-GP)")
    plt.show()

show_generated(G)
