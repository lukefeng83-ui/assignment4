import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from generator import generate_samples
from model import get_model
from trainer import train_gan


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0002


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    torch.manual_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    data_loader = DataLoader(
        mnist_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = get_model("GAN").to(device)
    criterion = nn.BCELoss()

    optimizers = {
        "generator": optim.Adam(
            model.generator.parameters(),
            lr=LEARNING_RATE,
            betas=(0.5, 0.999)
        ),
        "discriminator": optim.Adam(
            model.discriminator.parameters(),
            lr=LEARNING_RATE,
            betas=(0.5, 0.999)
        )
    }

    trained_model = train_gan(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizers,
        device=device,
        epochs=EPOCHS
    )

    torch.save(
        trained_model.generator.state_dict(),
        "mnist_generator.pth"
    )
    torch.save(
        trained_model.discriminator.state_dict(),
        "mnist_discriminator.pth"
    )

    print("Saved mnist_generator.pth")
    print("Saved mnist_discriminator.pth")

    generate_samples(
        trained_model,
        device=device,
        num_samples=10
    )


if __name__ == "__main__":
    main()
