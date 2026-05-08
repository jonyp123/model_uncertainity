"""Capacity-limited neural network used by the prototype."""

from __future__ import annotations

import torch
from torch import nn


class TinyPatternNet(nn.Module):
    """A deliberately small network with one observable hidden layer."""

    def __init__(self, hidden_size: int = 6) -> None:
        super().__init__()
        self.hidden = nn.Linear(3, hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        prediction, _ = self.predict_with_activation(inputs)
        return prediction

    def predict_with_activation(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        hidden_activation = self.activation(self.hidden(inputs))
        prediction = self.output(hidden_activation)
        return prediction, hidden_activation
