"""Train the original Module 6 GAN."""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model
from trainer import train_gan


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MNIST GAN.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = get_model("GAN").to(device)
    criterion = nn.BCELoss()
    optimizers = {
        "generator": optim.Adam(
            model.generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
        ),
        "discriminator": optim.Adam(
            model.discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
        ),
    }

    trained_model = train_gan(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizers,
        device=device,
        epochs=args.epochs,
    )
    torch.save(trained_model.generator.state_dict(), "mnist_generator.pth")
    torch.save(trained_model.discriminator.state_dict(), "mnist_discriminator.pth")
    print("Saved mnist_generator.pth and mnist_discriminator.pth")


if __name__ == "__main__":
    main()
