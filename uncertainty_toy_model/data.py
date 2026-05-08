"""Generated numeric pattern data for the uncertainty toy model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


PatternFn = Callable[[float], tuple[list[float], float]]


PATTERN_FUNCTIONS: dict[str, PatternFn] = {
    "Pattern A": lambda x: ([x, x + 1.0, x + 2.0], x + 3.0),
    "Pattern B": lambda x: ([x, 2.0 * x, 3.0 * x], 4.0 * x),
    "Pattern C": lambda x: ([x, x**2.0, x**3.0], x**4.0),
    "Pattern D": lambda x: ([x, x + 2.0, x + 4.0], x + 6.0),
}


@dataclass(frozen=True)
class PatternExample:
    pattern_name: str
    x: float
    raw_input: list[float]
    raw_target: float
    input_tensor: torch.Tensor
    target_tensor: torch.Tensor


class Normalizer:
    """Simple global scaler for stable toy training."""

    def __init__(self, input_scale: float = 30.0, output_scale: float = 100.0) -> None:
        self.input_scale = input_scale
        self.output_scale = output_scale

    def input(self, values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32) / self.input_scale

    def target(self, value: float) -> torch.Tensor:
        return torch.tensor([value / self.output_scale], dtype=torch.float32)

    def output_to_raw(self, value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().reshape(-1)[0].item())
        return value * self.output_scale


def make_example(pattern_name: str, x: float, normalizer: Normalizer) -> PatternExample:
    raw_input, raw_target = PATTERN_FUNCTIONS[pattern_name](x)
    return PatternExample(
        pattern_name=pattern_name,
        x=x,
        raw_input=raw_input,
        raw_target=raw_target,
        input_tensor=normalizer.input(raw_input),
        target_tensor=normalizer.target(raw_target),
    )


def generate_examples(
    pattern_names: list[str],
    x_values: list[float],
    normalizer: Normalizer | None = None,
) -> list[PatternExample]:
    normalizer = normalizer or Normalizer()
    return [
        make_example(pattern_name, x, normalizer)
        for pattern_name in pattern_names
        for x in x_values
    ]


def split_examples(
    train_patterns: list[str],
    all_patterns: list[str] | None = None,
    normalizer: Normalizer | None = None,
) -> tuple[list[PatternExample], list[PatternExample], Normalizer]:
    """Return a small initial train set and a broader test set."""

    normalizer = normalizer or Normalizer()
    all_patterns = all_patterns or list(PATTERN_FUNCTIONS)
    train = generate_examples(train_patterns, [1, 2, 3, 4, 5], normalizer)
    test = generate_examples(all_patterns, [1.5, 2.5, 3.5, 4.5], normalizer)
    return train, test, normalizer
