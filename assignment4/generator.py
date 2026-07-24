import math

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10):
    """
    Generate random latent vectors, run the Generator, and plot a grid.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    generator = model.generator if hasattr(model, "generator") else model
    latent_dim = getattr(model, "latent_dim", None)

    if latent_dim is None:
        latent_dim = getattr(generator, "latent_dim", 100)

    generator = generator.to(device)
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        generated_images = generator(noise).cpu()

    # Tanh outputs [-1, 1]; convert to [0, 1] for display.
    generated_images = ((generated_images + 1) / 2).clamp(0, 1)

    columns = min(5, num_samples)
    rows = math.ceil(num_samples / columns)

    grid = make_grid(
        generated_images,
        nrow=columns,
        padding=2
    )

    plt.figure(figsize=(columns * 2, rows * 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.title("GAN-generated MNIST samples")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return generated_images
