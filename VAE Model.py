import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import copy
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import os

IMG_SHAPE = [28, 28, 1]
IMG_SIZE = 28*28
INNER_SIZE = 600
LATENT_DIM = 50
LEARNING_RATE = 0.002
BATCH_SIZE = 100
N_EPOCHS = 30
DROP_PROB = 0.3
ALPHA = 0.01
TRAIN = True


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    DEVICE = 'cpu'
    print('CUDA is not available.  Training on CPU ...')
else:
    DEVICE = 'cuda'
    print('CUDA is available!  Training on GPU ...')

# Download the data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                 train=True, transform=transforms.ToTensor())
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                train=False, transform=transforms.ToTensor())
# load the data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False)

# define encoder architecture
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc12 = nn.Sequential(nn.Linear(IMG_SIZE, INNER_SIZE),
                                 nn.BatchNorm1d(INNER_SIZE),
                                 nn.Softplus(),
                                 nn.Dropout(DROP_PROB),
                                 nn.Linear(INNER_SIZE, INNER_SIZE),
                                 nn.BatchNorm1d(INNER_SIZE),
                                 nn.Softplus(),
                                 nn.Dropout(DROP_PROB))
        self.fc_mean = nn.Linear(INNER_SIZE, LATENT_DIM)
        self.fc_var = nn.Linear(INNER_SIZE, LATENT_DIM)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc12(x)
        log_z_var = self.fc_var(x)
        z_mean = self.fc_mean(x)
        # sample epsilon
        std = log_z_var.mul(0.5).exp_()
        epsilon = Variable(std.data.new(std.size()).normal_()).to(DEVICE)
        z = z_mean + std * epsilon
        return z, z_mean, log_z_var

# define decoder architecture
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.im_shape = IMG_SHAPE
        self.fc12_out = nn.Sequential(nn.Linear(LATENT_DIM, INNER_SIZE),
                                      nn.BatchNorm1d(INNER_SIZE),
                                      nn.Softplus(),
                                      nn.Dropout(DROP_PROB),
                                      nn.Linear(INNER_SIZE, INNER_SIZE),
                                      nn.BatchNorm1d(INNER_SIZE),
                                      nn.Softplus(),
                                      nn.Dropout(DROP_PROB),
                                      nn.Linear(INNER_SIZE, IMG_SIZE),
                                      nn.BatchNorm1d(IMG_SIZE),
                                      nn.Sigmoid())

    def forward(self, x):
        x = self.fc12_out(x)
        x = x.view(x.size(0), self.im_shape[2], self.im_shape[0], self.im_shape[1])
        return x

# define VAE architecture
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.im_shape = IMG_SHAPE
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

def train(trainloader, model, milestones, factor):
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1 / factor)

    for epoch in range(N_EPOCHS):
        running_loss = 0
        model.train()
        criterion = nn.BCELoss(reduction='sum')

        for images, labels in trainloader:
            optimizer.zero_grad()
            images = images.to(DEVICE)

            # Forward pass
            gener_x, z, z_mean, log_z_var = model(images)

            # define loss
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_z_var - z_mean.pow(2) - log_z_var.exp(), dim=1))
            images = images.view(images.size(0), -1)
            gener_x = gener_x.view(gener_x.size(0), -1)
            BCE_loss = IMG_SIZE * F.binary_cross_entropy(gener_x, images)
            loss = ALPHA * kl_loss + BCE_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()
        total_loss = running_loss / len(trainloader.dataset)

        print("Epoch: {}/{}.. ".format(epoch + 1, N_EPOCHS),
              "Training Loss: {:.3f}.. ".format(total_loss))

    # Save weights (fixed path)
    torch.save(model.state_dict(), 'VAE.pth')

if TRAIN == True:
    model = VAE()
    model = model.to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    train(trainloader, model, milestones=[20], factor=10)

# examine image generation
model = VAE()
model.load_state_dict(torch.load('VAE.pth'))
model.to(DEVICE)

n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
model.eval()
for j in range(n):
    for i in range(n):
        z_sample  = np.random.normal(size=LATENT_DIM).reshape(1, LATENT_DIM)
        z_sample  = torch.from_numpy(z_sample).float().to(DEVICE)
        x_gener = model.decoder(z_sample)
        x_gener = x_gener.cpu().data.numpy()
        digit = x_gener.reshape(digit_size, digit_size, IMG_SHAPE[2])

        d_x = i * digit_size
        d_y = j * digit_size
        figure[d_x:d_x + digit_size, d_y:d_y + digit_size] = digit[:, :, 0]

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.show()