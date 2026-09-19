from __future__ import annotations

import os
from typing import Callable, Iterable, Protocol, Tuple

import torch
import torch.nn as nn

from models import (
    Discriminator,
    Embedder,
    Generator,
    Recovery,
    Supervisor,
)


class _RNNClassType(Protocol):
    """Protocol for RNN types (e.g., nn.LSTM, nn.GRU) that can be used in the default TimeGAN initialization."""

    def __call__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        **kwargs,
    ) -> nn.Module: ...


class _OptimizerType(Protocol):
    """Protocol for optimizer types (e.g., torch.optim.Adam) that can be used for the TimeGAN model optimizers."""

    def __call__(self, params, lr: float, **kwargs) -> torch.optim.Optimizer: ...  # type: ignore


class TimeGAN:
    """Implements the TimeGAN architecture for generating synthetic time series data.

    The TimeGAN model consists of five main components:
    - Embedder: Encodes the input time series data into a latent representation.
    - Recovery: Decodes the latent representation back to the original data space, reconstructing the input time series.
    - Generator: Generates synthetic time series data from a latent representation.
    - Supervisor: Provides additional supervision to the generator by predicting the next time step in the latent space.
    - Discriminator: Distinguishes between real and synthetic latent representations to improve the quality of the generated data.
    They could be initialized with custom architectures, but the `default_init()` method provides a convenient way to set up all components with the same RNN-based architectures.

    The model is trained in three phases:
    1. Embedding Phase: Train only the Embedder-Recovery pair to encode and reconstruct the input data.
    2. Supervised Phase: Train the Supervisor specifically to predict the next time step in the latent space.
    3. Joint Training Phase: Train both the Generator and Discriminator as a whole.

    Arguments:
        noise_dim (int): The dimensionality of the noise vector input to the Generator.
        lr (float, optional): Learning rate for all optimizers (automatically created when components are initialized). Defaults to 0.001.
        gamma (float, optional): Hyperparameter for balancing the generator and discriminator losses. Defaults to 1.0.
        optimizer_class (_OptimizerType, optional): The optimizer class to use for all model components (automatically instantiated when components are initialized). Defaults to torch.optim.Adam. Must be a class that can be instantiated with `parameters` and `lr` arguments (e.g., torch.optim.Adam, torch.optim.SGD).

    Properties:
        models (nn.ModuleList): A list of all model components in the order of [embedder, recovery, generator, supervisor, discriminator]. Raises an error if any component is not initialized.
        noise_dim (int): The dimensionality of the noise vector input to the Generator.
        lr (float): Learning rate for all optimizers.
        gamma (float): Hyperparameter for balancing the generator and discriminator losses.
        PHASES (class): A nested class containing string constants representing the three different training phases of the TimeGAN model: "Embedding", "Supervised", and "Joint".
        embedder (nn.Module): The Embedder model component. Must be initialized before training or generation.
        recovery (nn.Module): The Recovery model component. Must be initialized before training or generation.
        generator (nn.Module): The Generator model component. Must be initialized before training or generation.
        supervisor (nn.Module): The Supervisor model component. Must be initialized before training or generation.
        discriminator (nn.Module): The Discriminator model component. Must be initialized before training or generation.
    """

    class PHASES:
        """Constants representing the different training phases of the TimeGAN model."""

        EMBEDDING: str = "Embedding"
        SUPERVISED: str = "Supervised"
        JOINT: str = "Joint"

    def __init__(
        self,
        noise_dim: int,
        lr: float = 0.001,
        gamma: float = 1.0,
        optimizer_class: _OptimizerType = torch.optim.Adam,  # type: ignore
    ):
        self.noise_dim = noise_dim
        self.embedder = self.recovery = self.generator = self.supervisor = (
            self.discriminator
        ) = None
        self.lr = lr
        self.gamma = gamma
        self.optimizer_class = optimizer_class

    def default_init(
        self, input_dim: int, latent_dim: int, rnn_type: _RNNClassType, num_layers: int
    ):
        """Convenience method to initialize all model components with default RNN-based architectures."""
        rnn_embedder = rnn_type(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        rnn_recovery = rnn_type(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        rnn_generator = rnn_type(
            input_size=self.noise_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        rnn_supervisor = rnn_type(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=max(
                1, num_layers - 1
            ),  # Matches original implementation where supervisor has one less layer than generator
            batch_first=True,
        )
        rnn_discriminator = rnn_type(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.embedder = Embedder(rnn_embedder, latent_dim)
        self.recovery = Recovery(rnn_recovery, latent_dim, input_dim)
        self.generator = Generator(rnn_generator, latent_dim)
        self.supervisor = Supervisor(rnn_supervisor, latent_dim)
        self.discriminator = Discriminator(rnn_discriminator, latent_dim)

    def __setattr__(self, name: str, value):
        """Override setattr to automatically create optimizers when model components are set."""
        super().__setattr__(name, value)
        _get_optimizer = lambda model: (
            self.optimizer_class(model.parameters(), lr=self.lr) if model else None  # type: ignore
        )
        match name:
            case "embedder":
                self._optimizer_embedder = _get_optimizer(value)
            case "recovery":
                self._optimizer_recovery = _get_optimizer(value)
            case "generator":
                self._optimizer_generator = _get_optimizer(value)
            case "supervisor":
                self._optimizer_supervisor = _get_optimizer(value)
            case "discriminator":
                self._optimizer_discriminator = _get_optimizer(value)

    @property
    def models(self):
        """The list of all model components. Raises an error if any component is not initialized.

        Returns:
            nn.ModuleList: A list of all initialized model components in the order of [Embedder, Recovery, Generator, Supervisor, Discriminator].
        """
        if (
            self.embedder is None
            or self.recovery is None
            or self.generator is None
            or self.supervisor is None
            or self.discriminator is None
        ):
            raise ValueError(
                "All model components must be initialized to access the models property. You can initialize default models with the `default_init()` method or set them manually."
            )
        return nn.ModuleList(
            [
                self.embedder,
                self.recovery,
                self.generator,
                self.supervisor,
                self.discriminator,
            ]
        )

    def train(
        self,
        data_loader: Iterable[torch.Tensor],
        iterations: int,
        device: str = "cpu",
        callback: Callable[[str, int, torch.Tensor], None] | None = None,
    ):
        """
        Train the TimeGAN model using the provided data loader and training loop.

        The training consists of three phases:
            1. Embedding Phase: Train the Embedder and Recovery to reconstruct the input data.
            2. Supervised Phase: Train the Supervisor to predict the next time step in the latent space.
            3. Joint Training Phase: Train the Generator and Discriminator together, while also fine-tuning the Embedder and Recovery.
        The `callback` function, if provided, will be called at the end of each iteration of each phase with the current phase name, iteration number, and loss value(s) for monitoring training progress.

        Arguments:
            data_loader (Iterable[torch.Tensor]): An iterable that yields batches of input time series data as tensors.
            iterations (int): The number of iterations to train for in each phase.
            device (str, optional): The device to train on (e.g., "cpu" or "cuda"). Defaults to "cpu".
            callback (Callable[[str, int, torch.Tensor], None], optional): A function that takes the current phase name, iteration number, and loss value(s) as arguments for monitoring training progress. Defaults to None.
        """
        assert self.models, "All model components must be initialized before training."

        draw_data = lambda: next(iter(data_loader)).to(
            device
        )  # assumes data_loader is an infinite generator
        _callback = lambda step_name, step_itr, loss: (
            callback(step_name, step_itr, loss) if callback else None
        )

        # Embedding and Recovery steps
        self.models.to(device).train()
        for embedder_iter in range(iterations):
            loss = self.embedding_step(draw_data())
            _callback(self.PHASES.EMBEDDING, embedder_iter, loss)

        # Generator and Supervisor steps
        self.embedder.eval()  # type: ignore
        for supervised_iter in range(iterations):
            loss = self.supervised_step(draw_data())
            _callback(self.PHASES.SUPERVISED, supervised_iter, loss)

        # Joint training steps (Generation and Discrimination)
        self.embedder.train()  # type: ignore
        for joint_iter in range(iterations):
            # Generation step
            # 2:1 ratio for generation:discrimination
            self.models.train()
            self.discriminator.eval()  # type: ignore
            gen_loss_1 = self.joint_step_generation(draw_data())
            gen_loss_2 = self.joint_step_generation(draw_data())

            # Discriminator step
            self.models.eval()
            self.discriminator.train()  # type: ignore
            disc_loss = self.joint_step_discrimination(draw_data())

            _callback(
                self.PHASES.JOINT, joint_iter, (gen_loss_1, gen_loss_2, disc_loss)
            )

    def embedding_step(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Runs through the embedding phase: maps input data to the latent space and back, and optimizes the reconstruction loss (if under training).

        Arguments:
            X (torch.Tensor): A batch of input time series data.
            training (bool, optional): Whether to perform backpropagation and optimization. Defaults to True.
        Returns:
            torch.Tensor: The reconstruction loss.
        """
        if (
            self.embedder is None
            or self.recovery is None
            or self._optimizer_embedder is None
            or self._optimizer_recovery is None
        ):
            raise ValueError(
                "Embedder, Recovery, and their optimizers must be initialized for the embedding step."
            )

        loss_fn = lambda x, y: 10 * torch.sqrt(nn.MSELoss()(x, y))
        self._optimizer_embedder.zero_grad()
        self._optimizer_recovery.zero_grad()

        H = self.embedder(X)
        X_tilde = self.recovery(H)
        E_loss0 = loss_fn(X_tilde, X)
        if training:
            E_loss0.backward()
            self._optimizer_embedder.step()
            self._optimizer_recovery.step()

        return E_loss0.detach()

    def supervised_step(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Runs through the supervised phase: ask the supervisor to predict the next time step in the latent space, and optimize the supervised loss (if under training).

        Arguments:
            X (torch.Tensor): A batch of input time series data.
            training (bool, optional): Whether to perform backpropagation and optimization. Defaults to True.
        Returns:
            torch.Tensor: The supervised loss."""
        if self.supervisor is None or self._optimizer_supervisor is None:
            raise ValueError(
                "Supervisor and its optimizer must be initialized for the supervised step."
            )
        elif self.embedder is None:
            raise ValueError("Embedder must be initialized for the supervised step.")

        loss_fn = nn.MSELoss()
        self._optimizer_supervisor.zero_grad()

        H = self.embedder(X)
        H_hat_supervise = self.supervisor(H)
        G_loss_S = loss_fn(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        if training:
            G_loss_S.backward()
            self._optimizer_supervisor.step()

        return G_loss_S.detach()

    def joint_step_generation(
        self, X: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs through the generation step of the joint training phase: trains the generator and supervisor to produce realistic synthetic data, while also fine-tuning the embedder and recovery. Optimizes the generator loss and embedding loss (if under training).

        Arguments:
            X (torch.Tensor): A batch of input time series data.
            training (bool, optional): Whether to perform backpropagation and optimization. Defaults to True.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The generator loss and embedding loss.
        """
        if (
            self.generator is None
            or self._optimizer_generator is None
            or self.supervisor is None
            or self._optimizer_supervisor is None
            or self.recovery is None
            or self._optimizer_recovery is None
            or self.embedder is None
            or self._optimizer_embedder is None
        ):
            raise ValueError(
                "Generator, Supervisor, Recovery, and Embedder and their optimizers must be initialized for the joint generation step."
            )
        if self.discriminator is None:
            raise ValueError(
                "Discriminator must be initialized for the joint generation step."
            )

        self._optimizer_generator.zero_grad()
        self._optimizer_supervisor.zero_grad()

        # Generator training
        input_noise = torch.randn(X.size(0), X.size(1), self.noise_dim, device=X.device)
        E_hat = self.generator(input_noise)
        H_hat = self.supervisor(E_hat)

        H = self.embedder(X)
        H_hat_supervise = self.supervisor(H)

        Y_fake = self.discriminator(H_hat)
        Y_fake_e = self.discriminator(E_hat)
        G_loss_U = torch.mean(
            torch.nn.BCEWithLogitsLoss()(Y_fake, torch.ones_like(Y_fake))
        )
        G_loss_U_e = torch.mean(
            torch.nn.BCEWithLogitsLoss()(Y_fake_e, torch.ones_like(Y_fake_e))
        )
        G_loss_S = torch.nn.MSELoss()(H[:, 1:, :], H_hat_supervise[:, :-1, :])

        X_hat = self.recovery(H_hat)
        G_loss_V1 = nn.L1Loss()(
            X_hat.std(
                dim=0, correction=0
            ),  # correction=0 to match the original code using tf.nn.moments
            X.std(dim=0, correction=0),
        )
        G_loss_V2 = nn.L1Loss()(X_hat.mean(dim=0), X.mean(dim=0))
        G_loss_V = G_loss_V1 + G_loss_V2

        # no idea where the 100 multiplier comes from
        G_loss = (
            G_loss_U
            + self.gamma * G_loss_U_e
            + 100 * torch.sqrt(G_loss_S)
            + 100 * G_loss_V
        )
        if training:
            G_loss.backward()
            self._optimizer_generator.step()
            self._optimizer_supervisor.step()

        # Embedder training
        self._optimizer_embedder.zero_grad()
        self._optimizer_recovery.zero_grad()

        H = self.embedder(X)
        X_tilde = self.recovery(H)
        E_loss_T0 = nn.MSELoss()(X_tilde, X)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)

        H_hat_supervise = self.supervisor(H)
        G_loss_S = nn.MSELoss()(H[:, 1:, :], H_hat_supervise[:, :-1, :])

        E_loss = E_loss0 + 0.1 * G_loss_S
        if training:
            E_loss.backward()
            self._optimizer_embedder.step()
            self._optimizer_recovery.step()

        return G_loss.detach(), E_loss.detach()

    def joint_step_discrimination(
        self, X: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """Runs through the discrimination step of the joint training phase: trains the discriminator to distinguish between real and synthetic latent representations, while keeping the generator and supervisor fixed. Optimizes the discriminator loss (if under training).

        Arguments:
            X (torch.Tensor): A batch of input time series data.
            training (bool, optional): Whether to perform backpropagation and optimization. Defaults to True.
        Returns:
            torch.Tensor: The discriminator loss.
        """
        if self.discriminator is None or self._optimizer_discriminator is None:
            raise ValueError(
                "Discriminator and its optimizer must be initialized for the joint discriminator step."
            )
        if self.embedder is None or self.generator is None or self.supervisor is None:
            raise ValueError(
                "Embedder, Generator, and Supervisor must be initialized for the joint discriminator step."
            )

        self._optimizer_discriminator.zero_grad()
        H = self.embedder(X)
        Y_real = self.discriminator(H)
        D_loss_real = torch.nn.BCEWithLogitsLoss()(Y_real, torch.ones_like(Y_real))

        Z = torch.randn(X.size(0), X.size(1), self.noise_dim, device=X.device)
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        Y_fake = self.discriminator(self.supervisor(H_hat))
        D_loss_fake = torch.nn.BCEWithLogitsLoss()(Y_fake, torch.zeros_like(Y_fake))

        Y_fake_e = self.discriminator(E_hat)
        D_loss_fake_e = torch.nn.BCEWithLogitsLoss()(
            Y_fake_e, torch.zeros_like(Y_fake_e)
        )

        D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
        if training:
            D_loss.backward()
            self._optimizer_discriminator.step()
        return D_loss.detach()

    def generate(
        self, num_samples: int, seq_len: int, device: str = "cpu"
    ) -> torch.Tensor:
        """Generates synthetic time series data using the trained Generator and Supervisor.

        Arguments:
            num_samples (int): The number of synthetic samples to generate.
            seq_len (int): The length of each generated time series sequence.
            device (str, optional): The device to perform generation on (e.g., "cpu" or "cuda"). Defaults to "cpu".
        Returns:
            torch.Tensor: A tensor of shape (num_samples, seq_len, feature_dim) containing the generated synthetic time series data.
        """
        if self.generator is None or self.supervisor is None or self.recovery is None:
            raise ValueError(
                "Generator, Supervisor, and Recovery must be initialized for data generation."
            )

        for model in [self.generator, self.supervisor, self.recovery]:
            model.eval().to(device)

        with torch.no_grad():
            Z = torch.randn(num_samples, seq_len, self.noise_dim, device=device)
            E_hat = self.generator(Z)
            H_hat = self.supervisor(E_hat)
            X_hat = self.recovery(H_hat)

        return X_hat

    class __ModelFiles:
        """Constants representing the filenames for saving and loading the model components and configuration."""

        EMBEDDER: str = "embedder.pth"
        RECOVERY: str = "recovery.pth"
        GENERATOR: str = "generator.pth"
        SUPERVISOR: str = "supervisor.pth"
        DISCRIMINATOR: str = "discriminator.pth"
        CONFIG: str = "config.pt"

    def load_weights(self, model_dir: str | os.PathLike):
        """Loads the model weights from the specified directory. Assumes that the model components have already been initialized (e.g., with `default_init()` or manually) and that the weights files are named according to the constants defined in `__ModelFiles`.

        Arguments:
            model_dir (str | os.PathLike): The directory path where the model weights are stored.
        """
        assert (
            self.models
        ), "All model components must be initialized before loading weights."

        self.embedder.load_state_dict(  # type: ignore
            torch.load(
                os.path.join(model_dir, self.__ModelFiles.EMBEDDER), weights_only=True
            )
        )
        self.recovery.load_state_dict(  # type: ignore
            torch.load(
                os.path.join(model_dir, self.__ModelFiles.RECOVERY), weights_only=True
            )
        )
        self.generator.load_state_dict(  # type: ignore
            torch.load(
                os.path.join(model_dir, self.__ModelFiles.GENERATOR), weights_only=True
            )
        )
        self.supervisor.load_state_dict(  # type: ignore
            torch.load(
                os.path.join(model_dir, self.__ModelFiles.SUPERVISOR), weights_only=True
            )
        )
        self.discriminator.load_state_dict(  # type: ignore
            torch.load(
                os.path.join(model_dir, self.__ModelFiles.DISCRIMINATOR),
                weights_only=True,
            )
        )

    def save(self, model_dir: str | os.PathLike):
        """Saves the model weights to the specified directory. Assumes that the model components have already been initialized (e.g., with `default_init()` or manually) and that the weights will be saved with filenames according to the constants defined in `__ModelFiles`.

        Arguments:
            model_dir (str | os.PathLike): The directory path where the model weights will be saved.
        """
        assert self.models, "All model components must be initialized before saving."

        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.embedder.state_dict(), os.path.join(model_dir, self.__ModelFiles.EMBEDDER))  # type: ignore
        torch.save(self.recovery.state_dict(), os.path.join(model_dir, self.__ModelFiles.RECOVERY))  # type: ignore
        torch.save(
            self.generator.state_dict(), os.path.join(model_dir, self.__ModelFiles.GENERATOR)  # type: ignore
        )
        torch.save(
            self.supervisor.state_dict(), os.path.join(model_dir, self.__ModelFiles.SUPERVISOR)  # type: ignore
        )
        torch.save(
            self.discriminator.state_dict(),  # type: ignore
            os.path.join(model_dir, self.__ModelFiles.DISCRIMINATOR),
        )
