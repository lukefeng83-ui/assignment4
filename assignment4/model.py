import torch
import torch.nn as nn

from cifar_models import SimpleDiffusionUNet, SmallEnergyModel


LATENT_DIM = 100


class Generator(nn.Module):
    """Generate 1 x 28 x 28 MNIST-like images from 100-dimensional noise."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(noise.size(0), 128, 7, 7)
        return self.network(x)


class Discriminator(nn.Module):
    """Classify a 1 x 28 x 28 image as real or fake."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        x = self.features(image)
        return self.classifier(x)


class GAN(nn.Module):
    """Container holding both the Generator and Discriminator."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, noise):
        return self.generator(noise)


class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
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
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


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
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


class VAE(nn.Module):
    """Small VAE included so get_model supports all requested model names."""

    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z).view(-1, 1, 28, 28)
        return reconstructed, mu, log_var


def get_model(model_name):
    """Return one of: FCNN, CNN, EnhancedCNN, VAE, GAN, EBM, or Diffusion."""

    model_map = {
        "FCNN": FCNN,
        "CNN": CNN,
        "ENHANCEDCNN": EnhancedCNN,
        "VAE": VAE,
        "GAN": GAN,
        "EBM": SmallEnergyModel,
        "ENERGYMODEL": SmallEnergyModel,
        "DIFFUSION": SimpleDiffusionUNet,
        "DIFFUSIONMODEL": SimpleDiffusionUNet,
    }

    normalized_name = model_name.replace("_", "").replace("-", "").upper()

    if normalized_name not in model_map:
        raise ValueError(
            "Invalid model name. Choose one of: "
            "FCNN, CNN, EnhancedCNN, VAE, GAN, EBM, Diffusion."
        )

    return model_map[normalized_name]()
