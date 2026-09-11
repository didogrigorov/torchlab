import torch

from torchlab import optimizers as O


def test_optimizer_registry_is_nonempty():
    registry = getattr(O, "OPTIMIZER_REGISTRY", {})
    assert registry, "OPTIMIZER_REGISTRY is missing or empty."


def test_lion_updates_parameter_in_gradient_direction():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = O.Lion([p], lr=0.1, weight_decay=0.0)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    assert p.item() < 1.0


def test_qhm_nu_zero_reduces_to_plain_gradient_step():
    p = torch.nn.Parameter(torch.tensor([1.0]))

    try:
        optimizer = O.QHM([p], lr=0.1, momentum=0.9, nu=0.0, weight_decay=0.0)
    except TypeError:
        # Some implementations name the momentum coefficient beta.
        optimizer = O.QHM([p], lr=0.1, beta=0.9, nu=0.0, weight_decay=0.0)

    p.grad = torch.tensor([2.0])
    optimizer.step()

    torch.testing.assert_close(
        p.detach(),
        torch.tensor([0.8]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fromage_one_step_is_finite():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = O.Fromage([p], lr=0.01)

    p.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert torch.isfinite(p).all()


def test_lion_can_train_small_regression_model():
    torch.manual_seed(42)

    x = torch.randn(128, 3)
    true_w = torch.tensor([[2.0], [-3.0], [0.5]])
    y = x @ true_w

    model = torch.nn.Linear(3, 1)
    optimizer = O.Lion(model.parameters(), lr=1e-2)

    initial_loss = None
    final_loss = None

    for _ in range(150):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)

        if initial_loss is None:
            initial_loss = loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss.detach().item()

    assert final_loss < initial_loss


def test_sam_two_step_contract():
    model = torch.nn.Linear(3, 1)
    base_optimizer = torch.optim.SGD

    try:
        optimizer = O.SAM(model.parameters(), base_optimizer, lr=0.01, rho=0.05)
    except TypeError:
        # Alternate constructor order used by some SAM implementations.
        optimizer = O.SAM(model.parameters(), base_optimizer=base_optimizer, lr=0.01, rho=0.05)

    x = torch.randn(8, 3)
    y = torch.randn(8, 1)

    first_loss = torch.mean((model(x) - y) ** 2)
    first_loss.backward()
    optimizer.first_step(zero_grad=True)

    second_loss = torch.mean((model(x) - y) ** 2)
    second_loss.backward()
    optimizer.second_step(zero_grad=True)

    assert all(torch.isfinite(p).all() for p in model.parameters())
