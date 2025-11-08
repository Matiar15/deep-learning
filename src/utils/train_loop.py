def train_loop(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    original_image = None
    reconstructed_image = None

    for idx, (images, noisy_images, _) in enumerate(train_loader):
        images, noisy_images = images.to(device), noisy_images.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_images)

        # Flatten target images to match the output shape
        # images_flat = images.view(images.size(0), -1)

        curr_loss = criterion(outputs, images)

        curr_loss.backward()
        optimizer.step()

        total_loss += curr_loss.item() * images.size(0)  # mnoży przez rozmiar batcha
        total_samples += images.size(0)

        if idx == 0:
            original_image = images[0]
            reconstructed_image = outputs[0]  # .view(original_image.size())

    avg_loss = total_loss / total_samples

    return avg_loss, original_image, reconstructed_image
