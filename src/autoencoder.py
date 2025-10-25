import torch.nn as nn

PICTURE_SIZE = 28 * 28

class AutoEncoder(nn.Module):
    def __init__(self, latent_space = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(PICTURE_SIZE, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, latent_space),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, PICTURE_SIZE),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
