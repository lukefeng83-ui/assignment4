"""Train the Assignment 4 diffusion model on CIFAR-10."""

import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model
from trainer import train_diffusion


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 diffusion model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--checkpoint", default="cifar10_diffusion.pth")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = get_model("Diffusion").to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    trained_model = train_diffusion(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        max_batches=args.max_batches,
    )

    torch.save(trained_model.state_dict(), args.checkpoint)
    print(f"Saved {args.checkpoint}")


if __name__ == "__main__":
    main()
