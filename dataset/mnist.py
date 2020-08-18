import torch
from torchvision import datasets, transforms



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./dataset/mnist', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=5, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./dataset/mnist', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)