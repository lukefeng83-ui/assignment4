import torch


def _get_optimizers(optimizer):
    """
    Accept either:
    - {"generator": opt_g, "discriminator": opt_d}
    - (opt_g, opt_d)
    """

    if isinstance(optimizer, dict):
        return optimizer["generator"], optimizer["discriminator"]

    if isinstance(optimizer, (tuple, list)) and len(optimizer) == 2:
        return optimizer[0], optimizer[1]

    raise TypeError(
        "optimizer must be a dictionary or a two-item tuple/list."
    )


def train_gan(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    """Train the Generator and Discriminator and return the trained GAN."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    model = model.to(device)
    generator = model.generator
    discriminator = model.discriminator

    generator_optimizer, discriminator_optimizer = _get_optimizers(
        optimizer
    )

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

            # 1. Train Discriminator
            discriminator_optimizer.zero_grad()

            real_predictions = discriminator(real_images)
            real_loss = criterion(real_predictions, real_labels)

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)

            fake_predictions = discriminator(fake_images.detach())
            fake_loss = criterion(fake_predictions, fake_labels)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # 2. Train Generator
            generator_optimizer.zero_grad()

            noise = torch.randn(batch_size, latent_dim, device=device)
            generated_images = generator(noise)
            predictions = discriminator(generated_images)

            generator_loss = criterion(predictions, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

            total_generator_loss += generator_loss.item()
            total_discriminator_loss += discriminator_loss.item()

        number_of_batches = len(data_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Generator Loss: "
            f"{total_generator_loss / number_of_batches:.4f} | "
            f"Discriminator Loss: "
            f"{total_discriminator_loss / number_of_batches:.4f}"
        )

    return model
