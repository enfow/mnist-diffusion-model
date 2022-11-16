from typing import Optional

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

MNIST_TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (1.0,)),
    ]
)


def generate_dataset():
    """Generate FashionMNIST train/test dataloaders

    Label
    -----

    label | desc
    ====================
    0     | T-shirt/top
    1     | Trouser
    2     | Pullover
    3     | Dress
    4     | Coat
    5     | Sandal
    6     | Shirt
    7     | Sneaker
    8     | Bag
    9     | Ankle boot

    FYI
    ---
    FashionMNIST: https://github.com/zalandoresearch/fashion-mnist
    """
    train_data = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=MNIST_TRANSFORM, download=True
    )
    test_data = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=MNIST_TRANSFORM, download=True
    )

    return train_data, test_data


def generate_dataloader(batch_size: int = 64):
    """Generate FashionMNIST train/test dataloaders

    DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    train_data, test_data = generate_dataset()

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, test_dataloader = generate_dataloader()

    for data, label in train_dataloader:
        print(f"train data  shape: {data.shape}")
        print(f"train label shape: {label.shape}")
        break
