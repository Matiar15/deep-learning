import numpy as np
import torch


class RandomPixels(object):
    def __init__(self, noise_ratio=0.1, seed=None):
        self.noise_ratio = noise_ratio
        self.seed = seed

    def __call__(self, tensor: torch.Tensor):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        noisy_tensor = tensor.clone()

        total_pixels = tensor.numel()

        num_pixels = int(self.noise_ratio * total_pixels)

        flat_tensor = noisy_tensor.view(-1)

        noise_indices = torch.randperm(flat_tensor.size(0))[:num_pixels]

        noise_values = torch.randint(0, 2, (num_pixels,), dtype=tensor.dtype)

        flat_tensor[noise_indices] = noise_values

        return noisy_tensor


def add_random_pixels(seed: int, image: np.ndarray, noise_ratio: float = 0.1):
    np.random.seed(seed)
    noisy_image = image.copy()

    num_pixels = int(noise_ratio * image.size)

    flat_image = noisy_image.flatten()
    noise_indices = np.random.choice(flat_image.size, num_pixels, replace=False)

    noise_values = np.random.choice([0.0, 1.0], num_pixels)

    flat_image[noise_indices] = noise_values

    return noisy_image
