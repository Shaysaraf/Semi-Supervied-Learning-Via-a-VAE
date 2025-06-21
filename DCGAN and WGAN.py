import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import time

# ========== Flags ==========
TRAIN_WGAN = True
TRAIN_DCGAN = True

# ========== Config ==========
SAVE_DIR = "models_combined"
os.makedirs(SAVE_DIR, exist_ok=True)

device     = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
image_size = 64
z_dim      = 64
nc         = 1
lr         = 0.0002
beta_1     = 0.5
beta_2     = 0.999
epochs     = 150
clip_value = 0.01

# ========== Dataset ==========
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataloader = DataLoader(
    FashionMNIST(".", download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

# ========== Models ==========
class Generator(nn.Module):
    def __init__(self, z_dim, nc=1, hidden_channels=64):
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.block(z_dim, hidden_channels * 8, 4, 1, 0),
            self.block(hidden_channels * 8, hidden_channels * 4),
            self.block(hidden_channels * 4, hidden_channels * 2),
            self.block(hidden_channels * 2, hidden_channels),
            nn.ConvTranspose2d(hidden_channels, nc, 4, 2, 1),
            nn.Tanh()
        )

    def block(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = x.view(len(x), self.z_dim, 1, 1)
        return self.gen(x)

class Critic(nn.Module):
    def __init__(self, nc=1, hidden_channels=64):
        super().__init__()
        self.crt = nn.Sequential(
            nn.Conv2d(nc, hidden_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self.block(hidden_channels, hidden_channels * 2),
            self.block(hidden_channels * 2, hidden_channels * 4),
            self.block(hidden_channels * 4, hidden_channels * 8),
            nn.Conv2d(hidden_channels * 8, 1, 4, 1, 0)
        )

    def block(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.crt(x).view(len(x), -1)

class Discriminator(nn.Module):
    def __init__(self, nc=1, hidden_channels=64):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(nc, hidden_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self.block(hidden_channels, hidden_channels * 2),
            self.block(hidden_channels * 2, hidden_channels * 4),
            self.block(hidden_channels * 4, hidden_channels * 8),
            nn.Conv2d(hidden_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def block(self, in_c, out_c, k=4, s=2, p=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.disc(x).view(-1)

# ========== Utilities ==========
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

def show_images(generator, z_dim, epoch, title="Generated Samples"):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, z_dim, device=device)
        samples = generator(noise).detach().cpu()
        grid = vutils.make_grid(samples, nrow=4, normalize=True)
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title(f"{title} (Epoch {epoch})")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()

def plot_losses(d_losses, g_losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss', color='blue')
    plt.plot(d_losses, label='Critic/Discriminator Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== WGAN Training ==========
if TRAIN_WGAN:
    print("\n Starting WGAN Training")
    gen = Generator(z_dim).to(device)
    crt = Critic(nc).to(device)
    gen.apply(weights_init)
    crt.apply(weights_init)

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    crt_opt = torch.optim.Adam(crt.parameters(), lr=lr, betas=(beta_1, beta_2))

    gen_losses, crt_losses = [], []

    for epoch in range(epochs):
        g_total, c_total = 0, 0
        for real, _ in tqdm(dataloader, desc=f"[WGAN Epoch {epoch+1}]"):
            real = real.to(device)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, z_dim, device=device)

            # Train Critic
            fake = gen(noise)
            crt_opt.zero_grad()
            loss_c = torch.mean(crt(fake.detach())) - torch.mean(crt(real))
            loss_c.backward()
            crt_opt.step()
            for p in crt.parameters():
                p.data.clamp_(-clip_value, clip_value)
            c_total += loss_c.item()

            # Train Generator
            gen_opt.zero_grad()
            fake = gen(noise)
            loss_g = -torch.mean(crt(fake))
            loss_g.backward()
            gen_opt.step()
            g_total += loss_g.item()

        crt_losses.append(c_total / len(dataloader))
        gen_losses.append(g_total / len(dataloader))
        print(f" Epoch {epoch+1}/{epochs} | Gen Loss: {gen_losses[-1]:.4f} | Critic Loss: {crt_losses[-1]:.4f}")
        show_images(gen, z_dim, epoch+1, title="WGAN Samples")

    plot_losses(crt_losses, gen_losses, "WGAN Losses")
    torch.save(gen.state_dict(), os.path.join(SAVE_DIR, "wgan_gen.pth"))
    torch.save(crt.state_dict(), os.path.join(SAVE_DIR, "wgan_crt.pth"))

# ========== DCGAN Training ==========
if TRAIN_DCGAN:
    print("\n Starting DCGAN Training")
    gen = Generator(z_dim).to(device)
    disc = Discriminator(nc).to(device)
    gen.apply(weights_init)
    disc.apply(weights_init)

    criterion = nn.BCELoss()
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    gen_losses, disc_losses = [], []

    for epoch in range(epochs):
        g_total, d_total = 0, 0
        for real, _ in tqdm(dataloader, desc=f"[DCGAN Epoch {epoch+1}]"):
            real = real.to(device)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, z_dim, device=device)
            fake = gen(noise)

            # Train Discriminator
            disc_opt.zero_grad()
            real_loss = criterion(disc(real), torch.ones(batch_size, device=device))
            fake_loss = criterion(disc(fake.detach()), torch.zeros(batch_size, device=device))
            loss_d = real_loss + fake_loss
            loss_d.backward()
            disc_opt.step()
            d_total += loss_d.item()

            # Train Generator
            gen_opt.zero_grad()
            fake = gen(noise)
            loss_g = criterion(disc(fake), torch.ones(batch_size, device=device))
            loss_g.backward()
            gen_opt.step()
            g_total += loss_g.item()

        disc_losses.append(d_total / len(dataloader))
        gen_losses.append(g_total / len(dataloader))
        print(f" Epoch {epoch+1}/{epochs} | Gen Loss: {gen_losses[-1]:.4f} | Disc Loss: {disc_losses[-1]:.4f}")
        show_images(gen, z_dim, epoch+1, title="DCGAN Samples")

    plot_losses(disc_losses, gen_losses, "DCGAN Losses")
    torch.save(gen.state_dict(), os.path.join(SAVE_DIR, "dcgan_gen.pth"))
    torch.save(disc.state_dict(), os.path.join(SAVE_DIR, "dcgan_disc.pth"))
