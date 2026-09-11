import inspect
import math

import torch
import torch.nn.functional as F

from torchlab.activations import adaptive as A


def assert_close(a, b, *, rtol=1e-6, atol=1e-6):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)


def test_coverage_entries_resolve_to_modules():
    coverage = getattr(A, "SECTION4_COVERAGE", {})
    assert coverage, "SECTION4_COVERAGE is missing or empty."

    missing = []
    for section, names in coverage.items():
        for name in names:
            if not hasattr(A, name):
                missing.append((section, name))
                continue

            obj = getattr(A, name)

            # Most coverage entries are nn.Module classes, but some entries such as
            # _msrf are helper functions used by a formula family.
            if inspect.isclass(obj):
                assert issubclass(obj, torch.nn.Module)
            else:
                assert callable(obj)

    assert not missing, missing


def test_all_section4_equations_are_mapped():
    equation_section = getattr(A, "EQUATION_SECTION", {})
    equation_coverage = getattr(A, "EQUATION_COVERAGE", {})

    assert set(equation_section) == set(range(255, 541))
    assert set(equation_coverage) == set(range(255, 541))


def test_swish_equation():
    x = torch.tensor([-1.0, -0.2, 0.0, 0.7, 2.0], dtype=torch.float64)
    beta = 1.4

    expected = x * torch.sigmoid(beta * x)
    assert_close(A.Swish(beta)(x), expected)


def test_celu_equation():
    x = torch.tensor([-1.0, 0.0, 0.3], dtype=torch.float64)
    alpha = 1.7

    expected = torch.where(
        x >= 0,
        x,
        alpha * torch.expm1(x / alpha),
    )

    assert_close(A.CELU(alpha)(x), expected)


def test_pelu_equation():
    x = torch.tensor([-1.0, 0.5], dtype=torch.float64)
    a, b = 2.0, 3.0

    expected = torch.where(
        x >= 0,
        (a / b) * x,
        a * torch.expm1(x / b),
    )

    assert_close(A.PELU(a, b)(x), expected)


def test_soft_exponential_identity_at_zero_parameter():
    x = torch.tensor([-0.7, 0.2, 1.0], dtype=torch.float64)
    module = A.SoftExponential(0.0)

    assert_close(module(x), x)


def test_square_plus_equation():
    x = torch.tensor([-0.5, 0.2, 1.0], dtype=torch.float64)
    eps = 0.04

    expected = 0.5 * (x + torch.sqrt(x * x + eps))
    assert_close(A.SquarePlus(eps, False)(x), expected)


def test_adaptive_activation_parameters_receive_gradient():
    x = torch.randn(32, requires_grad=True)
    module = A.Swish(1.2)

    y = module(x).mean()
    y.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    params = list(module.parameters())
    if params:
        assert all(p.grad is not None for p in params if p.requires_grad)
