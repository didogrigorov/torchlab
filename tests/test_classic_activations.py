import inspect
import math

import torch
import torch.nn.functional as F

from torchlab.activations import classic as A


def assert_close(a, b, *, rtol=1e-6, atol=1e-6):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)


def test_coverage_entries_resolve_to_modules():
    """Every class listed by the Section 3 coverage map should exist."""
    coverage = getattr(A, "SECTION3_COVERAGE", {})
    assert coverage, "SECTION3_COVERAGE is missing or empty."

    missing = []
    for section, names in coverage.items():
        for name in names:
            if not hasattr(A, name):
                missing.append((section, name))
                continue

            cls = getattr(A, name)
            assert inspect.isclass(cls)
            assert issubclass(cls, torch.nn.Module)

    assert not missing, missing


def test_gelu_matches_torch_reference():
    x = torch.linspace(-4, 4, 101, dtype=torch.float64)
    assert_close(A.GELU()(x), F.gelu(x))


def test_gelu_tanh_approximation():
    x = torch.linspace(-3, 3, 31, dtype=torch.float64)
    expected = 0.5 * x * (
        1
        + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))
        )
    )
    assert_close(A.GELUTanhApprox()(x), expected)


def test_gelu_sigmoid_approximation():
    x = torch.linspace(-3, 3, 31, dtype=torch.float64)
    expected = x * torch.sigmoid(1.702 * x)
    assert_close(A.GELUSigmoidApprox()(x), expected)


def test_hexpo_literal_equation():
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=torch.float64)
    a, b, c, d = 2.0, 3.0, 4.0, 5.0

    expected = torch.where(
        x >= 0,
        -a * (torch.exp(-x / b) - 1),
        c * (torch.exp(-x / d) - 1),
    )

    assert_close(A.Hexpo(a, b, c, d)(x), expected)


def test_sign_relu():
    x = torch.tensor([-2.0, -0.5, 0.0, 2.0], dtype=torch.float64)
    a = 0.3

    expected = torch.where(
        x >= 0,
        x,
        a * x / (x.abs() + 1),
    )

    assert_close(A.SignReLU(a)(x), expected)


def test_square_softmax_is_normalized():
    x = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float64)
    y = A.SquareSoftmax(4.0)(x)

    assert_close(y.sum(dim=-1), torch.ones(1, dtype=x.dtype))
    assert torch.all(y >= 0)


def test_classic_activation_backward_pass():
    x = torch.randn(16, dtype=torch.float64, requires_grad=True)
    y = A.GELU()(x).sum()
    y.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
