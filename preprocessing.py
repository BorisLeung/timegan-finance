import warnings

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing_extensions import Literal


def to_windows(
    sequence: np.ndarray, window_size: int, step_size: int = 1
) -> np.ndarray:
    """Convert a sequence into overlapping windows of specified size.

    Args:
        - sequence (np.ndarray): Input sequence data.
        - window_size (int): Size of each window.

    Returns:
        - np.ndarray: Array of windows.
    """
    seq_len, _ = sequence.shape
    max_start = seq_len - window_size

    return np.stack(
        [
            sequence[start : start + window_size]
            for start in range(0, max_start, step_size)
        ]
    )


class WindowScaler:
    """A class to scale windows of time series data using Min-Max scaling."""

    def __init__(self, clip: bool = True, feature_range: tuple[int, int] = (0, 1)):
        self.clip = clip
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, windows: np.ndarray):
        self.min_, self.max_ = np.min(windows, axis=(0, 1)), np.max(
            windows, axis=(0, 1)
        )

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        scaled = (windows - self.min_) / (self.max_ - self.min_)
        scaled = (
            scaled * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        if self.clip:
            scaled = np.clip(scaled, self.feature_range[0], self.feature_range[1])
        return scaled

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        self.fit(windows)
        return self.transform(windows)

    def inverse_transform(self, windows: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        unscaled = (windows - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        unscaled = unscaled * (self.max_ - self.min_) + self.min_
        return unscaled


# =======================================================================
# DEPRECATED: WindowProcessor class for TimeGAN preprocessing
# Note: This class is mixing scaling and window slicing together, which is not ideal. It is recommended to use the separate WindowScaler and to_windows functions instead.
# ======================================================================
class WindowProcessor:
    """DEPRECATED: A class to process time series data into windows for TimeGAN training."""

    def __warn_deprecated(self):
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "WindowProcessor is deprecated and will be removed in a future version. "
            "Please use other functions.",
            DeprecationWarning,
        )
        warnings.simplefilter("default", DeprecationWarning)

    def __init__(
        self,
        window_size: int,
        scaling: Literal["minmax", "standard", "none"] | None = None,
    ):
        self.__warn_deprecated()
        self.window_size = window_size
        assert scaling in (
            "minmax",
            "standard",
            "none",
            None,
        ), "scaling must be 'minmax', 'standard', or 'none'/None"
        self.scaling = scaling
        if self.scaling == "standard":
            self.scaler = StandardScaler()
        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

    def fit(self, sequence: np.ndarray):
        """Fit the scaler to the sequence data."""
        if self.scaler:
            self.scaler.fit(sequence)

    def transform(self, sequence: np.ndarray):
        """Transform the sequence data into windows, applying scaling if specified."""
        if self.scaler:
            sequence = self.scaler.transform(sequence)
        seq_len, _ = sequence.shape
        max_start = seq_len - self.window_size
        windows = []

        for start in range(max_start):
            window = sequence[start : start + self.window_size]
            windows.append(window)

        return np.stack(windows)

    def fit_transform(self, sequence: np.ndarray):
        """Fit the scaler and transform the sequence data."""
        self.fit(sequence)
        return self.transform(sequence)

    def inverse_transform(self, windows: np.ndarray):
        """Inverse transform the windows back to original scale."""
        if self.scaler:
            return np.stack(
                [self.scaler.inverse_transform(window) for window in windows]
            )
        else:
            return windows


# ========================================================================
# ADDITIONAL PREPROCESSING FUNCTIONS TO SUPPORT FINANCIAL TIME SERIES
#
# Note: in the original TimeGAN implementation, they did not use any
# specific preprocessing techniques on financial data; but in practice,
# it is both common and necessary to use log returns or percentage changes
# to stabilize the data. The following functions provide handy utilities
# for calculating and reversing log returns and percentage changes from
# 1D or 2D financial time series data.
# ========================================================================


def log_return(data: np.ndarray) -> np.ndarray:
    """Calculate log returns of 1 path (1D or 2D array).

    Args:
        - data (np.ndarray): Input data array.

    Returns:
        - np.ndarray: Log returns of the input data.
    """
    if len(data.shape) == 1:
        return np.diff(np.log(data))
    elif len(data.shape) == 2:
        return np.diff(np.log(data), axis=0)
    else:
        raise ValueError("Input data must be 1D or 2D array.")


def reverse_log_return(
    data: np.ndarray, initial_value: float | np.ndarray, return_terminal: bool = False
) -> np.ndarray:
    """Reverse log returns to get original values.

    Args:
        - data (np.ndarray): Log return data.
        - initial_value (float | np.ndarray): Initial value(s) to start the reconstruction.
        - return_terminal (bool): Whether to return the terminal value.
    Returns:
        - np.ndarray: Reconstructed original values.
    """
    if len(data.shape) == 1:
        assert isinstance(
            initial_value, (int, float)
        ), "initial_value must be a scalar for 1D data"
        if return_terminal:
            return initial_value * np.exp(np.sum(data))
        else:
            return np.concatenate(
                ([initial_value], initial_value * np.exp(np.cumsum(data)))
            )
    elif len(data.shape) == 2:
        num_features = data.shape[1]
        if return_terminal:
            return initial_value * np.exp(np.sum(data, axis=0))
        else:
            return np.concatenate(
                (
                    (
                        np.array([initial_value] * num_features).reshape(1, -1)
                        if isinstance(initial_value, (int, float))
                        else initial_value
                    ),
                    initial_value * np.exp(np.cumsum(data, axis=0)),
                ),
                axis=0,
            )
    else:
        raise ValueError("Input data must be 1D or 2D array.")


def pct_change(data: np.ndarray) -> np.ndarray:
    """Calculate percentage change of the data.

    Args:
        - data (np.ndarray): Input data array.

    Returns:
        - np.ndarray: Percentage change of the input data.
    """
    if len(data.shape) == 1:
        return np.diff(data) / data[:-1]
    elif len(data.shape) == 2:
        return np.diff(data, axis=0) / data[:-1, :]
    else:
        raise ValueError("Input data must be 1D or 2D array.")


def reverse_pct_change(
    data: np.ndarray, initial_value: float | np.ndarray
) -> np.ndarray:
    """Reverse percentage change to get original values.

    Args:
        - data (np.ndarray): Percentage change data.
        - initial_value (float | np.ndarray): Initial value(s) to start the reconstruction.

    Returns:
        - np.ndarray: Reconstructed original values.
    """
    if len(data.shape) == 1:
        assert isinstance(
            initial_value, (int, float)
        ), "initial_value must be a scalar for 1D data"
        return np.concatenate(([initial_value], initial_value * (1 + data).cumprod()))
    elif len(data.shape) == 2:
        num_features = data.shape[1]
        return np.concatenate(
            (
                (
                    np.array([initial_value] * num_features).reshape(1, -1)
                    if isinstance(initial_value, (int, float))
                    else initial_value
                ),
                initial_value * (1 + data).cumprod(axis=0),
            ),
            axis=0,
        )
    else:
        raise ValueError("Input data must be 1D or 2D array.")
