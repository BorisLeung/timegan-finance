from typing import Protocol

import torch.nn as nn


class _ActivationType(Protocol):
    """Protocol for activation function types that can be used in the TimeGAN models."""

    def __call__(self) -> nn.Module: ...


class Embedder(nn.Module):
    """Encodes the input time series data into a latent representation."""

    def __init__(
        self,
        rnn: nn.Module,
        hidden_dim: int,
        activation: _ActivationType = nn.Sigmoid,
    ):
        super().__init__()
        self.rnn = rnn
        self.hidden_dim = hidden_dim
        self.activation_fn = activation()
        self.dense = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_fn
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.dense(h)


class Recovery(nn.Module):
    """Decodes the latent representation back to the original data space, reconstructing the input time series."""

    def __init__(
        self,
        rnn: nn.Module,
        latent_dim: int,
        output_dim: int,
        activation: _ActivationType = nn.Sigmoid,
    ):
        super().__init__()
        self.rnn = rnn
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.activation_fn = activation()
        self.dense = nn.Sequential(
            nn.Linear(self.latent_dim, self.output_dim), self.activation_fn
        )

    def forward(self, h):
        r, _ = self.rnn(h)
        return self.dense(r)


class Generator(nn.Module):
    """Generates synthetic time series data from a latent representation."""

    def __init__(
        self,
        rnn: nn.Module,
        latent_dim: int,
        activation: _ActivationType = nn.Sigmoid,
    ):
        super().__init__()
        self.rnn = rnn
        self.latent_dim = latent_dim
        self.activation_fn = activation()
        self.dense = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), self.activation_fn
        )

    def forward(self, z):
        g, _ = self.rnn(z)
        return self.dense(g)


class Supervisor(nn.Module):
    """Predicts the next time step in the latent space, helping to capture temporal dependencies."""

    def __init__(
        self,
        rnn: nn.Module,
        latent_dim: int,
        activation: _ActivationType = nn.Sigmoid,
    ):
        super().__init__()
        self.rnn = rnn
        self.latent_dim = latent_dim
        self.activation_fn = activation()
        self.dense = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), self.activation_fn
        )

    def forward(self, h):
        s, _ = self.rnn(h)
        return self.dense(s)


class Discriminator(nn.Module):
    """Distinguishes between real and synthetic time series data in the latent space."""

    def __init__(self, rnn: nn.Module, latent_dim: int):
        super().__init__()
        self.rnn = rnn
        self.latent_dim = latent_dim
        self.dense = nn.Linear(self.latent_dim, 1)

    def forward(self, h):
        d, _ = self.rnn(h)
        return self.dense(d)
