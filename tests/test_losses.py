import torch

from torchlab import losses as L


def assert_close(a, b, *, rtol=1e-6, atol=1e-6):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)


def test_loss_registry_is_nonempty():
    registry = getattr(L, "LOSS_REGISTRY", {})
    assert registry, "LOSS_REGISTRY is missing or empty."


def test_dice_loss_perfect_prediction_is_near_zero():
    # Use probabilities explicitly when supported.
    pred = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    target = pred.clone()

    try:
        loss_fn = L.DiceLoss(from_logits=False)
    except TypeError:
        loss_fn = L.DiceLoss()

    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-5


def test_tversky_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 8, 8)).float()

    try:
        loss_fn = L.TverskyLoss(from_logits=True)
        loss = loss_fn(logits, target)
    except TypeError:
        loss_fn = L.TverskyLoss()
        loss = loss_fn(torch.sigmoid(logits), target)

    assert torch.isfinite(loss)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_rmse_matches_definition():
    pred = torch.tensor([1.0, 3.0, 6.0])
    target = torch.tensor([2.0, 1.0, 5.0])

    expected = torch.sqrt(torch.mean((pred - target) ** 2))
    actual = L.RMSELoss()(pred, target)

    assert_close(actual, expected)


def test_log_cosh_matches_definition():
    pred = torch.tensor([0.2, 1.3, -0.4], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64)
    error = pred - target

    expected = torch.log(torch.cosh(error)).mean()
    actual = L.LogCoshLoss()(pred, target)

    assert_close(actual, expected)


def test_contrastive_loss_zero_for_identical_positive_pair():
    x1 = torch.randn(8, 16)
    x2 = x1.clone()
    target = torch.ones(8)

    loss = L.ContrastiveLoss()(x1, x2, target)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-7


def test_barron_loss_backward():
    pred = torch.randn(32, requires_grad=True)
    target = torch.randn(32)

    loss = L.BarronLoss()(pred, target)
    assert torch.isfinite(loss)

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
