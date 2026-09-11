# TorchLab

TorchLab is a research-oriented PyTorch library collecting activation functions,
loss functions, and optimization algorithms that are useful for experimentation
with neural networks.

The project is intended to make a broad set of published methods available behind
a consistent PyTorch-style API.

## Package layout

```text
src/
└── torchlab/
    ├── __init__.py
    ├── activations/
    │   ├── __init__.py
    │   ├── classic.py
    │   └── adaptive.py
    ├── losses/
    │   ├── __init__.py
    │   └── losses.py
    └── optimizers/
        ├── __init__.py
        └── optimizers.py
```

## Installation for development

Create and activate a virtual environment, then install TorchLab in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
```

## Basic usage

```python
import torch

from torchlab.activations.classic import GELU
from torchlab.activations.adaptive import Swish
from torchlab.losses import DiceLoss
from torchlab.optimizers import Lion

x = torch.randn(8, 32)
y = GELU()(x)
z = Swish()(x)

print(y.shape, z.shape)
```

Optimizer example:

```python
import torch
from torchlab.optimizers import Lion

model = torch.nn.Linear(10, 1)
optimizer = Lion(model.parameters(), lr=1e-4)

x = torch.randn(32, 10)
target = torch.randn(32, 1)

prediction = model(x)
loss = torch.nn.functional.mse_loss(prediction, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Testing a source checkout

From the repository root:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

For a clean wheel test:

```bash
python -m build
python -m twine check --strict dist/*
python -m pip install --force-reinstall dist/torchlab-*.whl
```

## Documentation

Project documentation can be hosted with GitHub Pages from the `docs/` directory.

## Research-code note

TorchLab contains implementations of methods described in research literature.
Some algorithms require special training-loop behavior, parameter constraints, or
approximations. Consult the project documentation and mathematical audit before
using unfamiliar methods in production.

## License

TorchLab is released under the MIT License. See `LICENSE`.
