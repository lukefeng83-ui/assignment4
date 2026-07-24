"""FastAPI application for the Module 6 GAN and Assignment 4 generators."""

import io
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from torchvision.utils import make_grid, save_image

from model import get_model
from trainer import sample_diffusion, sample_ebm


BASE_DIR = Path(__file__).resolve().parent


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
GAN_PATH = BASE_DIR / "mnist_generator.pth"
DIFFUSION_PATH = BASE_DIR / "cifar10_diffusion.pth"
EBM_PATH = BASE_DIR / "cifar10_ebm.pth"

app = FastAPI(
    title="Assignment 4 Generative Models API",
    description=(
        "The original Module 6 MNIST GAN plus CIFAR-10 diffusion and "
        "energy-based image generators."
    ),
    version="4.0",
)


def _load_model(model_name: str, checkpoint_path: Path, generator_only: bool = False):
    if not checkpoint_path.exists():
        return None

    model = get_model(model_name).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if generator_only:
        model.generator.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model


gan_model = _load_model("GAN", GAN_PATH, generator_only=True)
diffusion_model = _load_model("Diffusion", DIFFUSION_PATH)
ebm_model = _load_model("EBM", EBM_PATH)


def _require_model(model, checkpoint_path: Path, command: str) -> None:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Checkpoint '{checkpoint_path.name}' was not found. "
                f"Train the model first with: {command}"
            ),
        )


def _images_to_png(images: torch.Tensor, filename: str) -> StreamingResponse:
    display_images = ((images.detach().cpu() + 1) / 2).clamp(0, 1)
    grid = make_grid(display_images, nrow=min(8, display_images.size(0)), padding=2)
    image_buffer = io.BytesIO()
    save_image(grid, image_buffer, format="PNG")
    image_buffer.seek(0)
    return StreamingResponse(
        image_buffer,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/")
def home():
    return {
        "message": "Assignment 4 API is running",
        "device": str(DEVICE),
        "endpoints": {
            "module_6_gan": "/generate?num_samples=10",
            "cifar10_diffusion": "/generate_with_diffusion?num_samples=10",
            "cifar10_ebm": "/generate_with_ebm?num_samples=10&sampling_steps=60",
            "documentation": "/docs",
        },
        "loaded_checkpoints": {
            "gan": gan_model is not None,
            "diffusion": diffusion_model is not None,
            "ebm": ebm_model is not None,
        },
    }


@app.get("/generate")
def generate_with_gan(num_samples: int = Query(default=10, ge=1, le=64)):
    _require_model(gan_model, GAN_PATH, "python train.py")
    with torch.no_grad():
        noise = torch.randn(num_samples, gan_model.latent_dim, device=DEVICE)
        images = gan_model.generator(noise)
    return _images_to_png(images, "gan_mnist.png")


@app.get("/generate_with_diffusion")
def generate_with_diffusion(
    num_samples: int = Query(default=10, ge=1, le=32),
):
    _require_model(
        diffusion_model, DIFFUSION_PATH, "python train_diffusion.py --epochs 5"
    )
    images = sample_diffusion(
        model=diffusion_model,
        num_samples=num_samples,
        device=DEVICE,
    )
    return _images_to_png(images, "diffusion_cifar10.png")


@app.get("/generate_with_ebm")
def generate_with_ebm(
    num_samples: int = Query(default=10, ge=1, le=32),
    sampling_steps: int = Query(default=60, ge=1, le=300),
    step_size: float = Query(default=10.0, gt=0.0, le=20.0),
    noise_scale: float = Query(default=0.01, ge=0.0, le=1.0),
):
    _require_model(ebm_model, EBM_PATH, "python train_ebm.py --epochs 5")
    images = sample_ebm(
        model=ebm_model,
        num_samples=num_samples,
        device=DEVICE,
        sampling_steps=sampling_steps,
        step_size=step_size,
        noise_scale=noise_scale,
    )
    return _images_to_png(images, "ebm_cifar10.png")
