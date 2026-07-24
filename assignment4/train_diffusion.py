"""Train a small DDPM-style diffusion model on CIFAR-10.

Example:
    python train_diffusion.py --epochs 1 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from cifar_models import GaussianDiffusion, SimpleDiffusionUNet, denormalize_cifar, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 diffusion model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--output", type=str, default="models/diffusion_cifar10.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = SimpleDiffusionUNet().to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (clean_images, _) in enumerate(loader):
            clean_images = clean_images.to(device)
            batch_size = clean_images.size(0)

            timesteps = torch.randint(0, args.timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(clean_images)
            noisy_images = diffusion.q_sample(clean_images, timesteps, noise)
            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"Batch {batch_idx}/{len(loader)} "
                    f"Noise prediction MSE {loss.item():.4f}"
                )

        print(f"Epoch {epoch + 1} average loss: {running_loss / len(loader):.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved diffusion weights to {output_path}")

    samples = diffusion.sample(model, num_samples=16, device=device, sampling_steps=50)
    save_image(denormalize_cifar(samples), "diffusion_cifar10_samples.png", nrow=4)
    print("Saved diffusion_cifar10_samples.png")


if __name__ == "__main__":
    main()
