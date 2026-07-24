"""Training and sampling utilities for GAN, diffusion and energy models."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Existing Module 6 GAN trainer
# ---------------------------------------------------------------------------
def _get_optimizers(optimizer):
    if isinstance(optimizer, dict):
        return optimizer["generator"], optimizer["discriminator"]
    if isinstance(optimizer, (tuple, list)) and len(optimizer) == 2:
        return optimizer[0], optimizer[1]
    raise TypeError("optimizer must be a dictionary or a two-item tuple/list")


def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    model = model.to(device)
    generator = model.generator
    discriminator = model.discriminator
    generator_optimizer, discriminator_optimizer = _get_optimizers(optimizer)
    latent_dim = getattr(model, "latent_dim", 100)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0

        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            discriminator_optimizer.zero_grad()
            real_loss = criterion(discriminator(real_images), real_labels)
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            generated_images = generator(noise)
            generator_loss = criterion(discriminator(generated_images), real_labels)
            generator_loss.backward()
            generator_optimizer.step()

            total_generator_loss += generator_loss.item()
            total_discriminator_loss += discriminator_loss.item()

        number_of_batches = max(len(data_loader), 1)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Generator Loss: {total_generator_loss / number_of_batches:.4f} | "
            f"Discriminator Loss: {total_discriminator_loss / number_of_batches:.4f}"
        )

    return model


# ---------------------------------------------------------------------------
# Diffusion utilities
# ---------------------------------------------------------------------------
@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_diffusion_schedule(
    timesteps: int = 100,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | str = "cpu",
) -> DiffusionSchedule:
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas, alphas, alpha_bars)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, image_shape) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.view(timesteps.size(0), *((1,) * (len(image_shape) - 1)))


def add_diffusion_noise(
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate x_t = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon."""

    if noise is None:
        noise = torch.randn_like(clean_images)
    signal_rates = _extract(schedule.alpha_bars.sqrt(), timesteps, clean_images.shape)
    noise_rates = _extract((1.0 - schedule.alpha_bars).sqrt(), timesteps, clean_images.shape)
    return signal_rates * clean_images + noise_rates * noise, noise


def train_diffusion(
    model,
    data_loader,
    optimizer,
    device="cpu",
    epochs: int = 5,
    timesteps: int | None = None,
    max_batches: int | None = None,
):
    """Train the UNet to predict the Gaussian noise added to CIFAR-10 images."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    model = model.to(device)
    number_of_timesteps = timesteps or getattr(model, "timesteps", 100)
    schedule = make_diffusion_schedule(number_of_timesteps, device=device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        processed_batches = 0

        for batch_index, (clean_images, _) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            clean_images = clean_images.to(device)
            batch_size = clean_images.size(0)
            sampled_timesteps = torch.randint(
                0, number_of_timesteps, (batch_size,), device=device, dtype=torch.long
            )
            noisy_images, target_noise = add_diffusion_noise(
                clean_images, sampled_timesteps, schedule
            )
            predicted_noise = model(noisy_images, sampled_timesteps)
            loss = F.mse_loss(predicted_noise, target_noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            processed_batches += 1

        average_loss = total_loss / max(processed_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs} | Diffusion Loss: {average_loss:.4f}")

    return model


@torch.no_grad()
def sample_diffusion(
    model,
    num_samples: int = 10,
    device="cpu",
    timesteps: int | None = None,
) -> torch.Tensor:
    """Generate CIFAR-10-sized images by reversing the diffusion process."""

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    model = model.to(device)
    model.eval()
    number_of_timesteps = timesteps or getattr(model, "timesteps", 100)
    schedule = make_diffusion_schedule(number_of_timesteps, device=device)
    channels = getattr(model, "image_channels", 3)
    image_size = getattr(model, "image_size", 32)

    images = torch.randn(num_samples, channels, image_size, image_size, device=device)

    for step in reversed(range(number_of_timesteps)):
        timestep_batch = torch.full((num_samples,), step, device=device, dtype=torch.long)
        predicted_noise = model(images, timestep_batch)

        beta_t = schedule.betas[step]
        alpha_t = schedule.alphas[step]
        alpha_bar_t = schedule.alpha_bars[step]

        model_mean = (1.0 / alpha_t.sqrt()) * (
            images - (beta_t / (1.0 - alpha_bar_t).sqrt()) * predicted_noise
        )

        if step > 0:
            posterior_noise = torch.randn_like(images)
            images = model_mean + beta_t.sqrt() * posterior_noise
        else:
            images = model_mean

        images.clamp_(-1.0, 1.0)

    return images


# ---------------------------------------------------------------------------
# Energy-based model utilities
# ---------------------------------------------------------------------------
def sample_ebm(
    model,
    num_samples: int = 10,
    device="cpu",
    sampling_steps: int = 60,
    step_size: float = 10.0,
    noise_scale: float = 0.01,
    initial_images: torch.Tensor | None = None,
    gradient_clip: float = 0.03,
) -> torch.Tensor:
    """Use Langevin dynamics to move images towards low-energy states.

    The neural-network parameters are frozen. ``requires_grad_(True)`` is placed
    on the input images because the gradient must be calculated with respect to
    the pixels rather than the model weights.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be at least 1")

    model = model.to(device)
    model.eval()
    channels = getattr(model, "image_channels", 3)
    image_size = getattr(model, "image_size", 32)

    if initial_images is None:
        images = torch.rand(num_samples, channels, image_size, image_size, device=device) * 2 - 1
    else:
        images = initial_images.detach().to(device)
        if images.size(0) != num_samples:
            raise ValueError("initial_images batch size must equal num_samples")

    previous_states = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    try:
        for _ in range(sampling_steps):
            with torch.no_grad():
                images = (images + torch.randn_like(images) * noise_scale).clamp(-1.0, 1.0)

            images = images.detach().requires_grad_(True)
            energy = model(images)
            gradients, = torch.autograd.grad(
                energy,
                images,
                grad_outputs=torch.ones_like(energy),
            )

            with torch.no_grad():
                gradients = gradients.clamp(-gradient_clip, gradient_clip)
                images = (images - step_size * gradients).clamp(-1.0, 1.0)
    finally:
        for parameter, old_state in zip(model.parameters(), previous_states):
            parameter.requires_grad_(old_state)

    return images.detach()


class SampleBuffer:
    """Persistent replay buffer used to reduce EBM mixing time."""

    def __init__(
        self,
        model,
        device="cpu",
        capacity: int = 8192,
        initial_size: int = 128,
    ):
        self.model = model
        self.device = device
        self.capacity = capacity
        channels = getattr(model, "image_channels", 3)
        image_size = getattr(model, "image_size", 32)
        self.examples = [
            torch.rand(1, channels, image_size, image_size, device=device) * 2 - 1
            for _ in range(initial_size)
        ]

    def sample_new_examples(
        self,
        batch_size: int,
        steps: int,
        step_size: float,
        noise_scale: float,
    ) -> torch.Tensor:
        channels = getattr(self.model, "image_channels", 3)
        image_size = getattr(self.model, "image_size", 32)
        number_new = max(1, int(round(batch_size * 0.05)))
        number_old = batch_size - number_new

        new_random = torch.rand(
            number_new, channels, image_size, image_size, device=self.device
        ) * 2 - 1
        old = (
            torch.cat(random.choices(self.examples, k=number_old), dim=0)
            if number_old > 0
            else new_random[:0]
        )
        starting_images = torch.cat([new_random, old], dim=0)

        generated = sample_ebm(
            self.model,
            num_samples=batch_size,
            device=self.device,
            sampling_steps=steps,
            step_size=step_size,
            noise_scale=noise_scale,
            initial_images=starting_images,
        )

        self.examples = list(torch.split(generated, 1, dim=0)) + self.examples
        self.examples = self.examples[: self.capacity]
        return generated


def train_ebm(
    model,
    data_loader,
    optimizer,
    device="cpu",
    epochs: int = 5,
    sampling_steps: int = 20,
    step_size: float = 10.0,
    noise_scale: float = 0.005,
    regularization: float = 0.1,
    max_batches: int | None = None,
):
    """Train an EBM with contrastive divergence and a persistent replay buffer."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    model = model.to(device)
    buffer = SampleBuffer(model, device=device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_real = 0.0
        total_fake = 0.0
        processed_batches = 0

        for batch_index, (real_images, _) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            real_images = real_images.to(device)
            real_images = (real_images + torch.randn_like(real_images) * noise_scale).clamp(-1.0, 1.0)
            batch_size = real_images.size(0)

            fake_images = buffer.sample_new_examples(
                batch_size=batch_size,
                steps=sampling_steps,
                step_size=step_size,
                noise_scale=noise_scale,
            )

            model.train()
            combined = torch.cat([real_images, fake_images.detach()], dim=0)
            energies = model(combined)
            real_energy, fake_energy = torch.split(
                energies, [batch_size, batch_size], dim=0
            )

            contrastive_divergence = real_energy.mean() - fake_energy.mean()
            regularization_loss = regularization * (
                real_energy.square().mean() + fake_energy.square().mean()
            )
            loss = contrastive_divergence + regularization_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += loss.item()
            total_real += real_energy.mean().item()
            total_fake += fake_energy.mean().item()
            processed_batches += 1

        count = max(processed_batches, 1)
        print(
            f"Epoch {epoch + 1}/{epochs} | EBM Loss: {total_loss / count:.4f} | "
            f"Real Energy: {total_real / count:.4f} | Fake Energy: {total_fake / count:.4f}"
        )

    return model
