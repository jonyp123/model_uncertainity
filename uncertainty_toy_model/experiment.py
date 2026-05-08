"""Training, evaluation, and active-learning style comparisons."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

import torch
from torch import nn

try:
    from .data import PATTERN_FUNCTIONS, Normalizer, PatternExample, generate_examples, make_example
    from .model import TinyPatternNet
    from .observer import ActivationObserver, UncertaintyExplanation
except ImportError:
    from data import PATTERN_FUNCTIONS, Normalizer, PatternExample, generate_examples, make_example
    from model import TinyPatternNet
    from observer import ActivationObserver, UncertaintyExplanation


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    predicted: int
    dont_know: int
    correct_predictions: int
    wrong_confident: int
    accuracy_on_predictions: float
    coverage: float


@dataclass(frozen=True)
class ExperimentResult:
    mode: str
    training_examples: int
    test_accuracy: float
    dont_know: int
    wrong_confident: int
    observer_paths: int


def train_model(
    model: TinyPatternNet,
    examples: list[PatternExample],
    epochs: int = 650,
    lr: float = 0.03,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        random.shuffle(examples)
        for example in examples:
            optimizer.zero_grad()
            prediction = model(example.input_tensor)
            loss = loss_fn(prediction.reshape(-1), example.target_tensor.reshape(-1))
            loss.backward()
            optimizer.step()


def build_observer(
    model: TinyPatternNet,
    examples: list[PatternExample],
    normalizer: Normalizer,
    raw_tolerance: float = 1.25,
) -> ActivationObserver:
    observer = ActivationObserver()
    model.eval()
    with torch.no_grad():
        for example in examples:
            prediction, activation = model.predict_with_activation(example.input_tensor)
            raw_prediction = normalizer.output_to_raw(prediction)
            if abs(raw_prediction - example.raw_target) <= raw_tolerance:
                observer.record(example.pattern_name, activation)
    return observer


def infer(
    model: TinyPatternNet,
    observer: ActivationObserver,
    example: PatternExample,
    threshold: float,
    normalizer: Normalizer,
) -> tuple[float, UncertaintyExplanation]:
    model.eval()
    with torch.no_grad():
        prediction, activation = model.predict_with_activation(example.input_tensor)
    explanation = observer.explain_uncertainty(activation, threshold)
    return normalizer.output_to_raw(prediction), explanation


def evaluate(
    model: TinyPatternNet,
    observer: ActivationObserver,
    examples: list[PatternExample],
    threshold: float,
    normalizer: Normalizer,
    raw_tolerance: float = 2.0,
) -> EvaluationResult:
    correct = 0
    wrong_confident = 0
    dont_know = 0
    predicted = 0

    for example in examples:
        raw_prediction, explanation = infer(model, observer, example, threshold, normalizer)
        if explanation.decision == "I don't know":
            dont_know += 1
            continue

        predicted += 1
        if abs(raw_prediction - example.raw_target) <= raw_tolerance:
            correct += 1
        else:
            wrong_confident += 1

    return EvaluationResult(
        total=len(examples),
        predicted=predicted,
        dont_know=dont_know,
        correct_predictions=correct,
        wrong_confident=wrong_confident,
        accuracy_on_predictions=correct / predicted if predicted else 0.0,
        coverage=predicted / len(examples) if examples else 0.0,
    )


def uncertain_patterns(
    model: TinyPatternNet,
    observer: ActivationObserver,
    examples: list[PatternExample],
    threshold: float,
    normalizer: Normalizer,
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        _, explanation = infer(model, observer, example, threshold, normalizer)
        if explanation.decision == "I don't know":
            counts[example.pattern_name] += 1
    return sorted(counts, key=counts.get, reverse=True)


def add_examples_for_pattern(
    training_examples: list[PatternExample],
    pattern_name: str,
    normalizer: Normalizer,
    step: int,
    count: int = 3,
) -> None:
    start = 6 + step * count
    for offset in range(count):
        x = float(start + offset)
        training_examples.append(make_example(pattern_name, x, normalizer))


def run_selection_experiment(
    mode: str,
    initial_patterns: list[str],
    all_patterns: list[str],
    normalizer: Normalizer,
    threshold: float = 0.92,
    max_rounds: int = 5,
    seed: int = 7,
) -> ExperimentResult:
    random.seed(seed)
    torch.manual_seed(seed)
    training_examples = generate_examples(initial_patterns, [1, 2, 3, 4, 5], normalizer)
    test_examples = generate_examples(all_patterns, [1.5, 2.5, 3.5, 4.5], normalizer)
    pool_patterns = list(PATTERN_FUNCTIONS)

    model = TinyPatternNet(hidden_size=6)
    final_observer = ActivationObserver()
    final_eval = None

    for round_index in range(max_rounds + 1):
        train_model(model, training_examples, epochs=500 if round_index == 0 else 260)
        final_observer = build_observer(model, training_examples, normalizer)
        final_eval = evaluate(model, final_observer, test_examples, threshold, normalizer)

        if final_eval.dont_know == 0:
            break
        if round_index == max_rounds:
            break

        uncertain = uncertain_patterns(model, final_observer, test_examples, threshold, normalizer)
        if mode == "uncertainty-guided" and uncertain:
            pattern_to_add = uncertain[0]
        else:
            pattern_to_add = random.choice(pool_patterns)
        add_examples_for_pattern(training_examples, pattern_to_add, normalizer, round_index)

    assert final_eval is not None
    return ExperimentResult(
        mode=mode,
        training_examples=len(training_examples),
        test_accuracy=final_eval.accuracy_on_predictions,
        dont_know=final_eval.dont_know,
        wrong_confident=final_eval.wrong_confident,
        observer_paths=final_observer.count(),
    )


def train_initial_demo(
    normalizer: Normalizer,
) -> tuple[TinyPatternNet, ActivationObserver, list[PatternExample]]:
    training_examples = generate_examples(["Pattern A", "Pattern B"], [1, 2, 3, 4, 5], normalizer)
    model = TinyPatternNet(hidden_size=6)
    train_model(model, training_examples)
    observer = build_observer(model, training_examples, normalizer)
    return model, observer, training_examples
