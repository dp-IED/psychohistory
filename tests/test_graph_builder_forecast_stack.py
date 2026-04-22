from __future__ import annotations

import torch

from baselines.france_plumbing_probes import _GATES_ORDERED
from baselines.graph_builder_forecast_stack import (
    BUILDER_EMBEDDING_DIM,
    FORECAST_GATE_ORDER,
    FORECAST_HEAD_IN_DIM,
    ForecastHead,
    GateMLP,
    forecast_brier_log_loss,
    path_a_head_input,
)


def test_gate_order_matches_france_plumbing_probes() -> None:
    assert FORECAST_GATE_ORDER == _GATES_ORDERED


def test_shapes_and_gates_in_unit_interval() -> None:
    B = 3
    bag = torch.randn(B, BUILDER_EMBEDDING_DIM)
    gate_mlp = GateMLP()
    gates = gate_mlp(bag)
    assert gates.shape == (B, 5)
    assert gates.min() >= 0.0 and gates.max() <= 1.0

    hi = path_a_head_input(bag, gates)
    assert hi.shape == (B, FORECAST_HEAD_IN_DIM)

    head = ForecastHead()
    p = head(hi)
    assert p.shape == (B, 1)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_loss_finite_and_backward_smoke() -> None:
    B = 4
    bag = torch.randn(B, BUILDER_EMBEDDING_DIM, requires_grad=True)
    y = torch.randint(0, 2, (B, 1), dtype=torch.float32)

    gate_mlp = GateMLP()
    head = ForecastHead()
    gates = gate_mlp(bag)
    p = head(path_a_head_input(bag, gates))
    loss = forecast_brier_log_loss(p, y)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert bag.grad is not None
    for param in list(gate_mlp.parameters()) + list(head.parameters()):
        assert param.grad is not None
