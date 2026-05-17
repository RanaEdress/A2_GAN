
import torch
import torch.nn as nn

from utils import (
    REQ_WEEKDAY_SIZE,
    REQ_MONTH_SIZE,
    REQ_LEAP_SIZE,
    REQ_DECADE_SIZE,
    GRID_SIZE,
)


class AEGridGenerator(nn.Module):
    def __init__(self, d_model=32, hidden_dim=128, latent_dim=64, grid_size=GRID_SIZE):
        super().__init__()

        self.weekday_embedding = nn.Embedding(REQ_WEEKDAY_SIZE, d_model)
        self.month_embedding = nn.Embedding(REQ_MONTH_SIZE, d_model)
        self.leap_embedding = nn.Embedding(REQ_LEAP_SIZE, d_model)
        self.decade_embedding = nn.Embedding(REQ_DECADE_SIZE, d_model)

        input_dim = 4 * d_model

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_size),
        )

    def embed_conditions(self, condition_features):
        weekday = self.weekday_embedding(condition_features[:, 0])
        month = self.month_embedding(condition_features[:, 1])
        leap = self.leap_embedding(condition_features[:, 2])
        decade = self.decade_embedding(condition_features[:, 3])
        return torch.cat([weekday, month, leap, decade], dim=1)

    def forward(self, condition_features):
        x = self.embed_conditions(condition_features)
        z = self.encoder(x)
        return self.decoder(z)


class GANGridGenerator(nn.Module):
    def __init__(self, noise_dim=32, d_model=32, hidden_dim=128, grid_size=GRID_SIZE):
        super().__init__()

        self.noise_dim = noise_dim

        self.weekday_embedding = nn.Embedding(REQ_WEEKDAY_SIZE, d_model)
        self.month_embedding = nn.Embedding(REQ_MONTH_SIZE, d_model)
        self.leap_embedding = nn.Embedding(REQ_LEAP_SIZE, d_model)
        self.decade_embedding = nn.Embedding(REQ_DECADE_SIZE, d_model)

        input_dim = 4 * d_model + noise_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, grid_size),
        )

    def embed_conditions(self, condition_features):
        weekday = self.weekday_embedding(condition_features[:, 0])
        month = self.month_embedding(condition_features[:, 1])
        leap = self.leap_embedding(condition_features[:, 2])
        decade = self.decade_embedding(condition_features[:, 3])
        return torch.cat([weekday, month, leap, decade], dim=1)

    def forward(self, condition_features):
        batch_size = condition_features.size(0)
        device = condition_features.device
        noise = torch.randn(batch_size, self.noise_dim, device=device)

        cond_emb = self.embed_conditions(condition_features)
        x = torch.cat([cond_emb, noise], dim=1)
        return self.net(x)


class CVAEGridGenerator(nn.Module):
    def __init__(self, d_model=32, hidden_dim=128, latent_dim=64, grid_size=GRID_SIZE):
        super().__init__()

        self.weekday_embedding = nn.Embedding(REQ_WEEKDAY_SIZE, d_model)
        self.month_embedding = nn.Embedding(REQ_MONTH_SIZE, d_model)
        self.leap_embedding = nn.Embedding(REQ_LEAP_SIZE, d_model)
        self.decade_embedding = nn.Embedding(REQ_DECADE_SIZE, d_model)

        input_dim = 4 * d_model

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_size),
        )

    def embed_conditions(self, condition_features):
        weekday = self.weekday_embedding(condition_features[:, 0])
        month = self.month_embedding(condition_features[:, 1])
        leap = self.leap_embedding(condition_features[:, 2])
        decade = self.decade_embedding(condition_features[:, 3])
        return torch.cat([weekday, month, leap, decade], dim=1)

    def encode(self, condition_features):
        cond_emb = self.embed_conditions(condition_features)
        h = self.encoder(cond_emb)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, cond_emb

    def forward(self, condition_features):
        mu, logvar, cond_emb = self.encode(condition_features)
        z = mu
        decoder_input = torch.cat([z, cond_emb], dim=1)
        return self.decoder(decoder_input)


class MLPGridGenerator(nn.Module):
    def __init__(self, d_model=32, hidden_dim=256, grid_size=GRID_SIZE, dropout=0.1):
        super().__init__()

        self.weekday_embedding = nn.Embedding(REQ_WEEKDAY_SIZE, d_model)
        self.month_embedding = nn.Embedding(REQ_MONTH_SIZE, d_model)
        self.leap_embedding = nn.Embedding(REQ_LEAP_SIZE, d_model)
        self.decade_embedding = nn.Embedding(REQ_DECADE_SIZE, d_model)

        input_dim = 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, grid_size),
        )

    def embed_conditions(self, condition_features):
        weekday = self.weekday_embedding(condition_features[:, 0])
        month = self.month_embedding(condition_features[:, 1])
        leap = self.leap_embedding(condition_features[:, 2])
        decade = self.decade_embedding(condition_features[:, 3])
        return torch.cat([weekday, month, leap, decade], dim=1)

    def forward(self, condition_features):
        x = self.embed_conditions(condition_features)
        return self.net(x)


def build_model(model_name: str):
    if model_name == "ae":
        return AEGridGenerator()

    if model_name == "gan":
        return GANGridGenerator()

    if model_name == "cvae":
        return CVAEGridGenerator()

    if model_name == "mlp":
        return MLPGridGenerator()

    raise ValueError(f"Unknown model name: {model_name}")
