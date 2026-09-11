# Changelog

All notable changes to TorchLab will be documented in this file.

The project follows Semantic Versioning where practical.

## [0.1.1] - 2026-09-11

### Changed

- Renamed the package/project to TorchLab.
- Organized activations, losses, and optimizers into dedicated subpackages.
- Added packaging metadata suitable for PyPI.
- Added GitHub Actions workflows for tests and PyPI Trusted Publishing.

### Mathematical corrections carried forward from the audited release

- Corrected VICReg variance/covariance conventions.
- Corrected Adan transformed-gradient recurrence.
- Corrected PNM and AdaPNM interleaved momentum recurrence.
- Corrected AdamP/SGDP projection handling and SGDP decay scaling.
- Corrected ASAM perturbation scaling.
- Aligned LAMB with the selected reference formulation.
- Preserved documented source ambiguities and finite approximations instead of
  silently replacing them with alternative formulas.

## [0.1.0]

### Added

- Classical activation-function collection.
- Adaptive activation-function collection.
- Research and non-core loss functions.
- Research and non-core optimizers.
- Mathematical audit material and tests.
