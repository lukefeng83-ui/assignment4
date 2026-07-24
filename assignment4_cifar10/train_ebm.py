"""Train the Assignment 4 energy-based model on CIFAR-10."""

import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model
from trainer import train_ebm


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 EBM")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=10.0)
    parser.add_argument("--noise-scale", type=float, default=0.005)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--checkpoint", default="cifar10_ebm.pth")
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

    model = get_model("EBM").to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.0, 0.999)
    )
    trained_model = train_ebm(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        sampling_steps=args.sampling_steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale,
        max_batches=args.max_batches,
    )

    torch.save(trained_model.state_dict(), args.checkpoint)
    print(f"Saved {args.checkpoint}")


if __name__ == "__main__":
    main()
