import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from datetime import datetime
import os

# Hyperparameters
IMG_SHAPE = [28, 28, 1]
IMG_SIZE = 28 * 28
INNER_SIZE = 600
LATENT_DIM = 50
LEARNING_RATE = 0.003
BATCH_SIZE = 100
N_EPOCHS = 30
DROP_PROB = 0.2
ALPHA = 0.05
TRAIN = True
MODEL_PATH = 'vae_checkpoint.pth'

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Data loading
train_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                  train=True, transform=transforms.ToTensor())
test_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                 train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc12 = nn.Sequential(
            nn.Linear(IMG_SIZE, INNER_SIZE),
            nn.BatchNorm1d(INNER_SIZE),
            nn.Softplus(),
            nn.Dropout(DROP_PROB),
            nn.Linear(INNER_SIZE, INNER_SIZE),
            nn.BatchNorm1d(INNER_SIZE),
            nn.Softplus(),
            nn.Dropout(DROP_PROB)
        )
        self.fc_mean = nn.Linear(INNER_SIZE, LATENT_DIM)
        self.fc_var = nn.Linear(INNER_SIZE, LATENT_DIM)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc12(x)
        log_z_var = self.fc_var(x)
        z_mean = self.fc_mean(x)
        std = log_z_var.mul(0.5).exp_()
        epsilon = Variable(std.data.new(std.size()).normal_()).to(DEVICE)
        z = z_mean + std * epsilon
        return z, z_mean, log_z_var

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc12_out = nn.Sequential(
            nn.Linear(LATENT_DIM, INNER_SIZE),
            nn.BatchNorm1d(INNER_SIZE),
            nn.Softplus(),
            nn.Dropout(DROP_PROB),
            nn.Linear(INNER_SIZE, INNER_SIZE),
            nn.BatchNorm1d(INNER_SIZE),
            nn.Softplus(),
            nn.Dropout(DROP_PROB),
            nn.Linear(INNER_SIZE, IMG_SIZE),
            nn.BatchNorm1d(IMG_SIZE),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc12_out(x)
        x = x.view(x.size(0), IMG_SHAPE[2], IMG_SHAPE[0], IMG_SHAPE[1])
        return x

# VAE Model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        z, z_mean, log_z_var = self.encoder(x)
        x_gener = self.decoder(z)
        return x_gener, z, z_mean, log_z_var

# Training
def train_model(model, optimizer, milestones, factor):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1/factor)
    criterion = nn.BCELoss(reduction='sum')

    for epoch in range(N_EPOCHS):
        running_loss = 0
        model.train()

        for images, _ in train_loader:
            images = images.to(DEVICE)
            optimizer.zero_grad()

            gener_x, z, z_mean, log_z_var = model(images)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_z_var - z_mean.pow(2) - log_z_var.exp(), dim=1))
            BCE_loss = IMG_SIZE * F.binary_cross_entropy(gener_x.view(gener_x.size(0), -1),
                                                         images.view(images.size(0), -1))
            loss = ALPHA * kl_loss + BCE_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        total_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{N_EPOCHS}, Training Loss: {total_loss:.3f}")

    # Save model and optimizer state
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Initialize model and optimizer
model = VAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train or load
if TRAIN:
    train_model(model, optimizer, milestones=[20], factor=10)
else:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {MODEL_PATH}")

# Generation
model.eval()
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

with torch.no_grad():
    for j in range(n):
        for i in range(n):
            z_sample = torch.randn(1, LATENT_DIM).to(DEVICE)
            x_gener = model.decoder(z_sample).cpu().numpy()
            digit = x_gener.reshape(digit_size, digit_size, IMG_SHAPE[2])
            d_x, d_y = i * digit_size, j * digit_size
            figure[d_x:d_x + digit_size, d_y:d_y + digit_size] = digit[:, :, 0]

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.axis('off')
plt.show()
