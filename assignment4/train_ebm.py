"""Train an Energy-Based Model on CIFAR-10.

Example:
    python train_ebm.py --epochs 1 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from cifar_models import SmallEnergyModel, denormalize_cifar, get_device, sample_ebm_langevin


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 EBM")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--langevin-steps", type=int, default=20)
    parser.add_argument("--output", type=str, default="models/ebm_cifar10.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = SmallEnergyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (real_images, _) in enumerate(loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            fake_images = sample_ebm_langevin(
                model=model,
                num_samples=batch_size,
                device=device,
                steps=args.langevin_steps,
            )

            real_energy = model(real_images).mean()
            fake_energy = model(fake_images.detach()).mean()

            # Lower energy for real CIFAR images, higher energy for generated negatives.
            contrastive_loss = real_energy - fake_energy
            regularization = 1e-3 * (real_energy.pow(2) + fake_energy.pow(2))
            loss = contrastive_loss + regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"Batch {batch_idx}/{len(loader)} "
                    f"Loss {loss.item():.4f} "
                    f"E(real) {real_energy.item():.4f} "
                    f"E(fake) {fake_energy.item():.4f}"
                )

        print(f"Epoch {epoch + 1} average loss: {running_loss / len(loader):.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved EBM weights to {output_path}")

    samples = sample_ebm_langevin(model, num_samples=16, device=device, steps=60)
    save_image(denormalize_cifar(samples), "ebm_cifar10_samples.png", nrow=4)
    print("Saved ebm_cifar10_samples.png")


if __name__ == "__main__":
    main()
