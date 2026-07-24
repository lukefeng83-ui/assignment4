"""Small CIFAR-10 Energy-Based Model and Diffusion Model utilities.

The models are intentionally lightweight so the FastAPI demo can run on a CPU
or a laptop while still demonstrating the training loops and input-gradient
sampling required by Assignment 4.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


CIFAR_IMAGE_CHANNELS = 3
CIFAR_IMAGE_SIZE = 32


def get_device() -> torch.device:
    """Return the best available PyTorch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize_cifar(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1] for saving/display."""

    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def images_to_png_bytes(images: torch.Tensor, nrow: Optional[int] = None) -> io.BytesIO:
    """Create an in-memory PNG grid from a batch of images."""

    images = images.detach().cpu()
    if images.min() < 0:
        images = denormalize_cifar(images)

    if nrow is None:
        nrow = min(8, max(1, images.shape[0]))

    grid = make_grid(images, nrow=nrow, padding=2)
    image_buffer = io.BytesIO()
    save_image(grid, image_buffer, format="PNG")
    image_buffer.seek(0)
    return image_buffer


class SmallEnergyModel(nn.Module):
    """A convolutional EBM that assigns one scalar energy to each CIFAR image."""

    def __init__(self, channels: int = CIFAR_IMAGE_CHANNELS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).view(-1)


@torch.enable_grad()
def sample_ebm_langevin(
    model: nn.Module,
    num_samples: int,
    device: torch.device,
    steps: int = 40,
    step_size: float = 0.08,
    noise_scale: float = 0.01,
    image_size: int = CIFAR_IMAGE_SIZE,
) -> torch.Tensor:
    """Sample low-energy images by taking gradients with respect to the input.

    This is the key EBM idea for the assignment: unlike ordinary network
    training, here the sampled image tensor requires gradients and is updated to
    reduce the model energy.
    """

    model.eval()
    x = torch.empty(num_samples, 3, image_size, image_size, device=device).uniform_(-1, 1)

    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        energy = model(x).sum()
        gradient = torch.autograd.grad(energy, x, create_graph=False)[0]
        x = x - step_size * gradient
        if noise_scale > 0:
            x = x + noise_scale * torch.randn_like(x)
        x = x.clamp(-1, 1)

    return x.detach()


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal timestep embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeResidualBlock(nn.Module):
    """A small residual block conditioned on a timestep embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class SimpleDiffusionUNet(nn.Module):
    """A compact U-Net that predicts noise for 32 x 32 CIFAR diffusion."""

    def __init__(self, channels: int = 3, base_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(channels, base_channels, kernel_size=3, padding=1)
        self.down_block = TimeResidualBlock(base_channels, base_channels, time_dim)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.middle_block = TimeResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up_block = TimeResidualBlock(base_channels * 2, base_channels, time_dim)
        self.output = nn.Conv2d(base_channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        h1 = self.stem(x)
        h1 = self.down_block(h1, time_embedding)
        h2 = self.downsample(h1)
        h2 = self.middle_block(h2, time_embedding)
        h3 = self.upsample(h2)
        h = torch.cat([h3, h1], dim=1)
        h = self.up_block(h, time_embedding)
        return self.output(h)


class GaussianDiffusion:
    """DDPM forward noising and reverse sampling utilities."""

    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: Optional[torch.device] = None,
    ):
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "GaussianDiffusion":
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, x_shape) -> torch.Tensor:
        out = values.gather(0, timesteps)
        return out.view(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean images at timestep t."""

        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = torch.sqrt(self._extract(self.alpha_bars, timesteps, x_start.shape))
        sqrt_one_minus_alpha_bar = torch.sqrt(
            1.0 - self._extract(self.alpha_bars, timesteps, x_start.shape)
        )
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """One reverse denoising step using the predicted noise."""

        beta_t = self._extract(self.betas, timesteps, x.shape)
        alpha_t = self._extract(self.alphas, timesteps, x.shape)
        alpha_bar_t = self._extract(self.alpha_bars, timesteps, x.shape)

        predicted_noise = model(x, timesteps)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )

        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().view(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        num_samples: int,
        device: torch.device,
        image_size: int = CIFAR_IMAGE_SIZE,
        sampling_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate CIFAR-sized samples from random noise."""

        model.eval()
        self.to(device)
        x = torch.randn(num_samples, 3, image_size, image_size, device=device)

        if sampling_steps is None or sampling_steps >= self.timesteps:
            time_indices = list(range(self.timesteps - 1, -1, -1))
        else:
            time_indices = torch.linspace(
                self.timesteps - 1,
                0,
                sampling_steps,
                dtype=torch.long,
            ).tolist()

        for time_index in time_indices:
            t = torch.full((num_samples,), int(time_index), device=device, dtype=torch.long)
            x = self.p_sample(model, x, t).clamp(-1, 1)

        return x


def load_weights_if_available(model: nn.Module, path: str | Path, device: torch.device) -> bool:
    """Load model weights if a checkpoint exists; return whether loading happened."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return False

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return True
