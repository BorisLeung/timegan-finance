# timegan-finance

A slightly extended PyTorch version of [TimeGAN](https://papers.nips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) (adapted from the original TensorFlow implementation: https://github.com/jsyoon0823/TimeGAN) that is primarily tailored for financial time-series.

## Features

- Lightweight serializable `TimeGAN` class with default RNN-based models (GRU/LSTM).
- Highly customizable components that support individual network replacements.
- Utilities for data preprocessing, including sliding window segmentation and return transformations.
- Callback support for progress reporting and custom logging.

## Quick Start

Install requirements:

```bash
pip install -r requirements.txt
```

Example usage:

```python
import torch
import torch.nn as nn
from timegan import TimeGAN

# create default model
timegan_ = TimeGAN(noise_dim=24)
timegan_.default_init(input_dim=6, latent_dim=64, rnn_type=nn.GRU, num_layers=3)

# train TimeGAN
timegan_.train(data_loader, iterations=1000, device='cpu', callback=my_callback)

# generate synthetic samples
synthetic = timegan_.generate(num_samples=100, seq_len=24, device='cpu')
```

See [example](./example.ipynb) for an end-to-end application.

## `TimeGAN` Class Key APIs

- `default_init(feature_dim, latent_dim, rnn_type, num_layers)` — auto-create all TimeGAN components of the same `rnn_type` with `num_layers`.
- `train(data_loader, iterations, device='cpu', callback=None)` — runs the three training phases. `callback(phase, iteration, loss)` is invoked each step.
- `generate(num_samples, seq_len, device='cpu')` — returns `torch.Tensor` of synthetic sequences.
- `save(path)` / `load(path)` — serializes and loads the trained TimeGAN model.
- `models` — returns a list of all submodels; raises if not initialized.
- `PHASES` — nested class with string constants: `EMBEDDING`, `SUPERVISED`, `JOINT`.
