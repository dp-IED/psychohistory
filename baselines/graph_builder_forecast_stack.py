"""Stage 2 forecast stack: gated bag representation → calibrated probability.

Gate vector layout matches ``_GATES_ORDERED`` in ``baselines/france_plumbing_probes.py`` (same
order as ``AssumptionEmphasis`` used there): index ``i`` is ``FORECAST_GATE_ORDER[i]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from schemas.graph_builder_probe import AssumptionEmphasis
from schemas.graph_builder_retrieval import BUILDER_EMBEDDING_DIM

FORECAST_GATE_ORDER: tuple[AssumptionEmphasis, ...] = (
    AssumptionEmphasis.PERSISTENCE,
    AssumptionEmphasis.PROPAGATION,
    AssumptionEmphasis.PRECURSOR,
    AssumptionEmphasis.SUPPRESSION,
    AssumptionEmphasis.COORDINATION,
)
FORECAST_NUM_GATES = len(FORECAST_GATE_ORDER)
FORECAST_HEAD_IN_DIM = BUILDER_EMBEDDING_DIM + FORECAST_NUM_GATES


class GateMLP(nn.Module):
    """Maps bag embedding to five soft gates in ``(0, 1)``, one per ``FORECAST_GATE_ORDER`` slot."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BUILDER_EMBEDDING_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, FORECAST_NUM_GATES),
            nn.Sigmoid(),
        )

    def forward(self, bag_repr: torch.Tensor) -> torch.Tensor:
        return self.net(bag_repr)


class ForecastHead(nn.Module):
    """Consumes ``[B, FORECAST_HEAD_IN_DIM]`` (bag + gates); returns ``p`` with shape ``[B, 1]``."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FORECAST_HEAD_IN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, head_input: torch.Tensor) -> torch.Tensor:
        return self.net(head_input)


def path_a_head_input(bag_repr: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    return torch.cat([bag_repr, gates], dim=-1)


def forecast_brier_log_loss(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    brier = 0.5 * ((p - y) ** 2).mean()
    log_term = -(y * torch.log(p + eps) + (1.0 - y) * torch.log(1.0 - p + eps)).mean()
    return brier + 0.5 * log_term


__all__ = [
    "BUILDER_EMBEDDING_DIM",
    "FORECAST_GATE_ORDER",
    "FORECAST_HEAD_IN_DIM",
    "FORECAST_NUM_GATES",
    "ForecastHead",
    "GateMLP",
    "forecast_brier_log_loss",
    "path_a_head_input",
]
