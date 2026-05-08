"""Activation-path observer for similarity-based uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class UncertaintyExplanation:
    best_match: str | None
    similarity: float
    threshold: float
    crossed_threshold: bool
    decision: str


class ActivationObserver:
    """Stores hidden activations from successful examples and compares new paths."""

    def __init__(self) -> None:
        self.activations_by_pattern: dict[str, list[torch.Tensor]] = {}

    def record(self, pattern_name: str, activation_vector: torch.Tensor) -> None:
        vector = activation_vector.detach().float().reshape(-1).cpu()
        self.activations_by_pattern.setdefault(pattern_name, []).append(vector)

    def similarity(self, activation_vector: torch.Tensor) -> tuple[str | None, float]:
        if not self.activations_by_pattern:
            return None, -1.0

        query = activation_vector.detach().float().reshape(1, -1).cpu()
        best_pattern: str | None = None
        best_score = -1.0

        for pattern_name, vectors in self.activations_by_pattern.items():
            stored = torch.stack(vectors)
            scores = F.cosine_similarity(query, stored, dim=1)
            pattern_score = float(scores.max().item())
            if pattern_score > best_score:
                best_pattern = pattern_name
                best_score = pattern_score

        return best_pattern, best_score

    def is_known(self, activation_vector: torch.Tensor, threshold: float) -> bool:
        _, score = self.similarity(activation_vector)
        return score >= threshold

    def explain_uncertainty(
        self,
        activation_vector: torch.Tensor,
        threshold: float = 0.92,
    ) -> UncertaintyExplanation:
        best_match, score = self.similarity(activation_vector)
        crossed = score >= threshold
        return UncertaintyExplanation(
            best_match=best_match,
            similarity=score,
            threshold=threshold,
            crossed_threshold=crossed,
            decision="predict" if crossed else "I don't know",
        )

    def count(self) -> int:
        return sum(len(vectors) for vectors in self.activations_by_pattern.values())
