"""Fast checks for CIFAR-10 model shapes and gradient control."""

import torch

from model import get_model
from trainer import add_diffusion_noise, make_diffusion_schedule, sample_diffusion, sample_ebm


def main() -> None:
    torch.manual_seed(0)

    diffusion = get_model("Diffusion")
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0, 10], dtype=torch.long)
    schedule = make_diffusion_schedule(diffusion.timesteps)
    noisy_images, target_noise = add_diffusion_noise(images, timesteps, schedule)
    predicted_noise = diffusion(noisy_images, timesteps)
    assert predicted_noise.shape == target_noise.shape == (2, 3, 32, 32)

    tiny_diffusion = get_model("Diffusion")
    tiny_diffusion.timesteps = 2
    generated_diffusion = sample_diffusion(
        tiny_diffusion, num_samples=2, timesteps=2
    )
    assert generated_diffusion.shape == (2, 3, 32, 32)

    ebm = get_model("EBM")
    energies = ebm(images)
    assert energies.shape == (2,)
    generated_ebm = sample_ebm(ebm, num_samples=2, sampling_steps=2)
    assert generated_ebm.shape == (2, 3, 32, 32)

    print("All CIFAR-10 smoke tests passed")


if __name__ == "__main__":
    main()
