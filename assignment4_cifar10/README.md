# Assignment 4: CIFAR-10 Diffusion and Energy-Based Models

This project extends the Module 6 FastAPI project by adding two image generators
trained on **CIFAR-10**:

- a time-conditioned UNet diffusion model;
- an energy-based model sampled with Langevin dynamics.

The original Module 6 MNIST GAN endpoint is retained.

## Main files

```text
model.py              model definitions
trainer.py            training and sampling loops
generator.py          image-grid helpers
train.py              original Module 6 MNIST GAN training
train_diffusion.py    CIFAR-10 diffusion training
train_ebm.py          CIFAR-10 EBM training
main.py               FastAPI application
smoke_test.py         fast model and gradient checks
Assignment4_Answers.md
```

## 1. Install dependencies

```bash
uv sync
```

## 2. Run a quick test

```bash
uv run python smoke_test.py
```

## 3. Train the CIFAR-10 models

Quick functional tests:

```bash
uv run python train_diffusion.py --epochs 1 --max-batches 5
uv run python train_ebm.py --epochs 1 --sampling-steps 2 --max-batches 2
```

Fuller training runs:

```bash
uv run python train_diffusion.py --epochs 5
uv run python train_ebm.py --epochs 5 --sampling-steps 20
```

The scripts create:

```text
cifar10_diffusion.pth
cifar10_ebm.pth
```

The EBM is slower because each training batch runs several gradient-based
Langevin sampling steps.

## 4. Start FastAPI

```bash
uv run fastapi dev main.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

Test:

```text
GET /generate_with_diffusion?num_samples=10
GET /generate_with_ebm?num_samples=10&sampling_steps=60
```

## Important EBM gradient step

During Langevin sampling, the model parameters are frozen and the image tensor
is marked for gradient tracking:

```python
images = images.detach().requires_grad_(True)
energy = model(images)
gradients, = torch.autograd.grad(
    energy,
    images,
    grad_outputs=torch.ones_like(energy),
)
images = images - step_size * gradients
```

This updates the pixels to lower the model's energy rather than updating the
neural-network parameters.
