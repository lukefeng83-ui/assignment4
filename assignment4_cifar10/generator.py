"""Visualisation helpers for the Module 6 GAN and Assignment 4 models."""

import math

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from trainer import sample_diffusion, sample_ebm


def _plot_image_grid(images: torch.Tensor, title: str) -> None:
    display_images = ((images.detach().cpu() + 1) / 2).clamp(0, 1)
    num_samples = display_images.size(0)
    columns = min(5, num_samples)
    rows = math.ceil(num_samples / columns)
    grid = make_grid(display_images, nrow=columns, padding=2)

    plt.figure(figsize=(columns * 2, rows * 2))
    image = grid.permute(1, 2, 0)
    if image.shape[-1] == 1:
        plt.imshow(image.squeeze(-1), cmap="gray")
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def generate_samples(model, device, num_samples: int = 10) -> torch.Tensor:
    generator = model.generator if hasattr(model, "generator") else model
    latent_dim = getattr(model, "latent_dim", getattr(generator, "latent_dim", 100))
    generator = generator.to(device)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        generated_images = generator(noise).cpu()
    _plot_image_grid(generated_images, "GAN-generated MNIST samples")
    return generated_images


def generate_diffusion_samples(model, device, num_samples: int = 10) -> torch.Tensor:
    generated_images = sample_diffusion(model, num_samples, device)
    _plot_image_grid(generated_images, "Diffusion-generated CIFAR-10 samples")
    return generated_images


def generate_ebm_samples(
    model,
    device,
    num_samples: int = 10,
    sampling_steps: int = 60,
) -> torch.Tensor:
    generated_images = sample_ebm(
        model,
        num_samples=num_samples,
        device=device,
        sampling_steps=sampling_steps,
    )
    _plot_image_grid(generated_images, "EBM-generated CIFAR-10 samples")
    return generated_images
