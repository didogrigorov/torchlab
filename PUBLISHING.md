# Publishing TorchLab

This checklist assumes the repository uses the `src/torchlab/` package layout.

## 1. Edit project metadata

Before publishing, open `pyproject.toml` and replace:

```text
YOUR_USERNAME
```

with your GitHub username.

Also review:

- package version;
- author/maintainer information;
- description;
- repository URLs;
- Python version support;
- PyTorch dependency floor.

## 2. Create a clean virtual environment

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## 3. Install development dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 4. Run the tests

```bash
pytest -q
```

Do not publish if unexpected tests fail.

## 5. Build distributions

Remove old build artifacts first:

```bash
python -m build
```

This should create a wheel and source archive under `dist/`.

## 6. Validate metadata

```bash
python -m twine check --strict dist/*
```

Both artifacts should pass.

## 7. Test the wheel locally

Create a separate virtual environment and install the built wheel:

```bash
python -m venv wheel-test
```

Activate it, then:

```bash
python -m pip install dist/torchlab-*.whl
```

Test imports:

```python
import torchlab
from torchlab.activations.classic import GELU
from torchlab.activations.adaptive import Swish
from torchlab.losses import DiceLoss
from torchlab.optimizers import Lion
```

## 8. TestPyPI (recommended before production)

Create an account at TestPyPI and upload:

```bash
python -m twine upload --repository testpypi dist/*
```

Then install from TestPyPI in a clean environment. PyTorch itself may need to
come from the normal PyPI index.

## 9. Push the repository to GitHub

```bash
git init
git add .
git commit -m "Initial TorchLab release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/torchlab.git
git push -u origin main
```

Check that the `Tests` GitHub Actions workflow succeeds.

## 10. Configure PyPI Trusted Publishing

Recommended production setup:

1. Create/verify your account on PyPI.
2. Enable two-factor authentication.
3. Create a GitHub environment named `pypi`.
4. On PyPI, configure a GitHub Trusted Publisher with:
   - owner: your GitHub username/organization;
   - repository: `torchlab`;
   - workflow: `release.yml`;
   - environment: `pypi`.
5. Optionally require manual approval for the `pypi` GitHub environment.

The included release workflow uses GitHub OIDC and does not require a permanent
PyPI token in GitHub Secrets.

## 11. Publish a release

Make sure the version in `pyproject.toml` is correct.

Commit and push:

```bash
git add .
git commit -m "Release 0.1.1"
git push
```

Create and publish a GitHub Release tagged:

```text
v0.1.1
```

The `release.yml` workflow will run tests, build distributions, validate them,
and publish them to PyPI through Trusted Publishing.

## 12. Verify the production package

In a fresh environment:

```bash
python -m pip install torchlab
```

Then test:

```python
import torchlab
from torchlab.activations.classic import GELU
from torchlab.losses import DiceLoss
from torchlab.optimizers import Lion
```

## Version rule

Published PyPI distribution files are immutable. If `0.1.1` has a bug, publish
a new version such as `0.1.2`; do not attempt to overwrite `0.1.1`.
