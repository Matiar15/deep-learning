import torch


def test_model_on(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, noisy_images, _ in dataloader:
            images, noisy_images = images.to(device), noisy_images.to(device)

            outputs = model(noisy_images)

            # Flatten target images to match the output shape
            images_flat = images.view(images.size(0), -1)

            curr_loss = criterion(outputs, images_flat)

            # Accumulate loss properly weighted by batch size
            total_loss += curr_loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples

        return avg_loss