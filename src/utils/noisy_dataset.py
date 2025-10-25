import copy

import torch
import typing

from torch.utils.data import Dataset

class NoisyDataset(Dataset):
    """
    Dataset wrapper który zwraca parę: (oryginał, zaszumiony obraz)

    Args:
        dataset: Bazowy dataset (np. FashionMNIST)
        noise_transform: Transformacja do dodania szumu (np. RandomPixels)
    """
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    def __init__(
            self,
            dataset: Dataset,
            noise_transform: typing.Callable
    ):
        self.dataset = dataset
        noisy_dataset = copy.deepcopy(dataset)
        noisy_dataset.transform = noise_transform
        self.noisy_dataset = noisy_dataset

    def __len__(self) -> int:
        return len(self.dataset) # type: ignore

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, typing.Any]:
        image, label = self.dataset[idx]
        noisy_image = self.noisy_dataset[idx][0]
        return image, noisy_image, label
