# Assignment 4 - CIFAR-10 EBM and Diffusion API

## Overview

This project extends the Module 6 / Assignment 3 FastAPI generative-model API.
It keeps the original MNIST GAN endpoint and adds two CIFAR-10 image generators:

1. **Energy-Based Model (EBM)** trained on CIFAR-10.
2. **Diffusion Model (DDPM-style)** trained on CIFAR-10.

The API can be deployed with Docker and queried through `/docs`.

## Files

```text
model.py                  # Model registry: GAN, EBM, Diffusion, etc.
cifar_models.py           # CIFAR-10 EBM, diffusion U-Net, DDPM utilities, samplers
train_ebm.py              # Train the Energy-Based Model on CIFAR-10
train_diffusion.py        # Train the diffusion model on CIFAR-10
main.py                   # FastAPI server with GAN, EBM, and diffusion endpoints
check_cifar10.py          # Quick smoke test
Dockerfile                # Docker deployment
requirements.txt          # Python dependencies
```

## Model Architecture

### Energy-Based Model

The EBM is a small convolutional network that maps each CIFAR-10 image
`(3, 32, 32)` to one scalar energy value. Real images are trained to have lower
energy than negative samples. Negative samples are generated with Langevin
dynamics by taking gradients with respect to the input image tensor.

### Diffusion Model

The diffusion model uses a compact time-conditioned U-Net. The forward process
adds Gaussian noise to CIFAR-10 images. The neural network learns to predict the
added noise, and sampling starts from random noise and repeatedly denoises it.

## Install Locally

```bash
pip install -r requirements.txt
```

## Quick Smoke Test

```bash
python check_cifar10.py
```

## Train the Models

Train the EBM:

```bash
python train_ebm.py --epochs 1 --batch-size 64
```

Train the diffusion model:

```bash
python train_diffusion.py --epochs 1 --batch-size 64
```

Training saves:

```text
models/ebm_cifar10.pth
models/diffusion_cifar10.pth
```

The API will still start without these files, using randomly initialized models,
so the Docker deployment and endpoints can be tested before long training runs.

## Run the API Locally

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

```text
GET /health
GET /generate?model=diffusion&num_samples=10&steps=50
GET /generate/diffusion?num_samples=10&steps=50
GET /generate/ebm?num_samples=10&steps=40
GET /generate/gan?num_samples=10
```

Each generation endpoint returns a PNG image grid.

## Docker Deployment

Build the image:

```bash
docker build -t assignment4-generative-api .
```

Run the container:

```bash
docker run -p 8000:8000 assignment4-generative-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## GitHub Submission

After adding the files, commit and push:

```bash
git add .
git commit -m "Add CIFAR-10 EBM and diffusion API for Assignment 4"
git push
```
