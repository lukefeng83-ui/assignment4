"""Model definitions for the Module 6 API and Assignment 4.

The original MNIST GAN from Module 6 is retained. Assignment 4 adds two
CIFAR-10 generators:

* ``DiffusionModel``: a small time-conditioned UNet that predicts noise.
* ``EnergyModel``: a convolutional network that outputs one scalar energy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_DIM = 100
DIFFUSION_TIMESTEPS = 100
CIFAR10_IMAGE_SIZE = 32
CIFAR10_CHANNELS = 3


# ---------------------------------------------------------------------------
# Module 6 models (kept so the original API endpoint continues to work)
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """Generate 1 x 28 x 28 MNIST-like images from latent noise."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = self.fc(noise)
        x = x.view(noise.size(0), 128, 7, 7)
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image))


class GAN(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.generator(noise)


class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z).view(-1, 1, 28, 28)
        return reconstructed, mu, log_var


# ---------------------------------------------------------------------------
# Assignment 4: CIFAR-10 diffusion model
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Encode integer timesteps using sinusoidal frequencies.

    For an embedding dimension ``d`` and pair index ``i``:

    sin(t / max_period ** (2i / d)), cos(t / max_period ** (2i / d)).
    """

    def __init__(self, dimension: int = 64, max_period: float = 10_000.0):
        super().__init__()
        if dimension < 4 or dimension % 2 != 0:
            raise ValueError("dimension must be an even number greater than or equal to 4")
        self.dimension = dimension
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float().view(-1)
        half = self.dimension // 2
        exponents = torch.arange(half, device=timesteps.device, dtype=torch.float32)
        frequencies = torch.exp(-math.log(self.max_period) * exponents / half)
        angles = timesteps[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class TimeResidualBlock(nn.Module):
    """Residual convolution block conditioned on a timestep embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = x + self.time_projection(time_embedding)[:, :, None, None]
        x = self.norm2(self.conv2(F.silu(x)))
        return F.silu(x + residual)


class DiffusionModel(nn.Module):
    """Compact UNet that predicts Gaussian noise for 3 x 32 x 32 images."""

    def __init__(
        self,
        timesteps: int = DIFFUSION_TIMESTEPS,
        time_dim: int = 64,
        image_channels: int = CIFAR10_CHANNELS,
        image_size: int = CIFAR10_IMAGE_SIZE,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.image_channels = image_channels
        self.image_size = image_size

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(image_channels, 32, kernel_size=3, padding=1)

        self.down_block1 = TimeResidualBlock(32, 32, time_dim)
        self.downsample1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

        self.down_block2 = TimeResidualBlock(64, 64, time_dim)
        self.downsample2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.middle1 = TimeResidualBlock(128, 128, time_dim)
        self.middle2 = TimeResidualBlock(128, 128, time_dim)

        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up_block1 = TimeResidualBlock(128, 64, time_dim)

        self.upsample2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up_block2 = TimeResidualBlock(64, 32, time_dim)

        self.output = nn.Conv2d(32, image_channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)

        x = self.input_conv(noisy_images)
        skip1 = self.down_block1(x, time_embedding)

        x = self.downsample1(skip1)
        skip2 = self.down_block2(x, time_embedding)

        x = self.downsample2(skip2)
        x = self.middle1(x, time_embedding)
        x = self.middle2(x, time_embedding)

        x = self.upsample1(x)
        x = self.up_block1(torch.cat([x, skip2], dim=1), time_embedding)

        x = self.upsample2(x)
        x = self.up_block2(torch.cat([x, skip1], dim=1), time_embedding)

        return self.output(x)


# ---------------------------------------------------------------------------
# Assignment 4: CIFAR-10 energy-based model
# ---------------------------------------------------------------------------
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """Assign one unnormalised scalar energy to each CIFAR-10 image."""

    def __init__(self, image_channels: int = CIFAR10_CHANNELS):
        super().__init__()
        self.image_channels = image_channels
        self.image_size = CIFAR10_IMAGE_SIZE
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = swish(self.conv1(images))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x).squeeze(-1)


EBM = EnergyModel


def get_model(model_name: str) -> nn.Module:
    """Return a model by name."""

    model_map = {
        "FCNN": FCNN,
        "CNN": CNN,
        "ENHANCEDCNN": EnhancedCNN,
        "VAE": VAE,
        "GAN": GAN,
        "DIFFUSION": DiffusionModel,
        "DIFFUSIONMODEL": DiffusionModel,
        "EBM": EnergyModel,
        "ENERGYMODEL": EnergyModel,
        "ENERGYBASEDMODEL": EnergyModel,
    }
    normalized_name = model_name.replace("_", "").replace("-", "").replace(" ", "").upper()
    if normalized_name not in model_map:
        raise ValueError(
            "Invalid model name. Choose one of: FCNN, CNN, EnhancedCNN, "
            "VAE, GAN, Diffusion, EBM."
        )
    return model_map[normalized_name]()
