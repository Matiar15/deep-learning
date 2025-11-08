import torch
import torch.nn as nn


class VAEAutoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )

        self.flattened_size = 64 * 7 * 7

        self.mean_layer = nn.Linear(self.flattened_size, latent_dim)
        self.log_var_layer = nn.Linear(self.flattened_size, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _reparameterization(mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x):
        batch = x.size(0)
        h = self.encoder(x)
        h_flat = h.view(batch, -1)
        mean = self.mean_layer(h_flat)
        log_var = self.log_var_layer(h_flat)
        return mean, log_var

    def decode(self, z):
        batch = z.size(0)
        h = self.fc_decode(z)
        h = h.view(batch, 64, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self._reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
