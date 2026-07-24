"""Quick smoke test for Assignment 4 models and samplers.

Run:
    python check_cifar10.py
"""

import torch

from cifar_models import GaussianDiffusion, SimpleDiffusionUNet, SmallEnergyModel, sample_ebm_langevin


def main():
    device = torch.device("cpu")

    ebm = SmallEnergyModel().to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    energy = ebm(x)
    assert energy.shape == (2,), f"Unexpected EBM output shape: {energy.shape}"

    ebm_samples = sample_ebm_langevin(ebm, num_samples=2, device=device, steps=2)
    assert ebm_samples.shape == (2, 3, 32, 32)

    diffusion_model = SimpleDiffusionUNet().to(device)
    diffusion = GaussianDiffusion(timesteps=10, device=device)
    t = torch.randint(0, 10, (2,), device=device)
    noisy = diffusion.q_sample(x, t)
    predicted_noise = diffusion_model(noisy, t)
    assert predicted_noise.shape == x.shape

    diffusion_samples = diffusion.sample(
        diffusion_model,
        num_samples=2,
        device=device,
        sampling_steps=2,
    )
    assert diffusion_samples.shape == (2, 3, 32, 32)

    print("Assignment 4 CIFAR-10 EBM and diffusion smoke test passed.")


if __name__ == "__main__":
    main()
