"""Run the full uncertainty-guided learning toy demo."""

from __future__ import annotations

try:
    from .data import Normalizer, generate_examples
    from .experiment import evaluate, infer, run_selection_experiment, train_initial_demo
    from .utils import format_float, print_table, set_seed
except ImportError:
    from data import Normalizer, generate_examples
    from experiment import evaluate, infer, run_selection_experiment, train_initial_demo
    from utils import format_float, print_table, set_seed


def show_inference_examples(threshold: float = 0.92) -> None:
    normalizer = Normalizer()
    model, observer, _ = train_initial_demo(normalizer)
    examples = generate_examples(["Pattern A", "Pattern B", "Pattern C", "Pattern D"], [2], normalizer)

    print("\nActivation-path decisions after training only on Pattern A and Pattern B:\n")
    for example in examples:
        prediction, explanation = infer(model, observer, example, threshold, normalizer)
        print(f"Input: {example.raw_input}")
        if explanation.decision == "predict":
            print(f"Prediction: {format_float(prediction)}")
        else:
            print(f"Prediction blocked: {format_float(prediction)}")
        print(f"Decision: {explanation.decision}")
        print(f"Best match: {explanation.best_match}")
        print(f"Similarity: {explanation.similarity:.2f}")
        print(f"Target: {format_float(example.raw_target)}")
        print()

    test_examples = generate_examples(["Pattern A", "Pattern B", "Pattern C", "Pattern D"], [1.5, 2.5, 3.5, 4.5], normalizer)
    result = evaluate(model, observer, test_examples, threshold, normalizer)
    print("Initial evaluation:")
    print_table(
        [
            {
                "Predicted": result.predicted,
                "I don't know": result.dont_know,
                "Accuracy": f"{result.accuracy_on_predictions:.2f}",
                "Wrong confident": result.wrong_confident,
            }
        ],
        ["Predicted", "I don't know", "Accuracy", "Wrong confident"],
    )


def compare_selection_modes(threshold: float = 0.92) -> None:
    normalizer = Normalizer()
    all_patterns = ["Pattern A", "Pattern B", "Pattern C", "Pattern D"]
    initial_patterns = ["Pattern A", "Pattern B"]
    random_result = run_selection_experiment(
        "random",
        initial_patterns,
        all_patterns,
        normalizer,
        threshold=threshold,
        seed=11,
    )
    guided_result = run_selection_experiment(
        "uncertainty-guided",
        initial_patterns,
        all_patterns,
        normalizer,
        threshold=threshold,
        seed=11,
    )

    print("\nSelection mode comparison:")
    print_table(
        [
            {
                "Mode": random_result.mode,
                "Training examples": random_result.training_examples,
                "Test accuracy": f"{random_result.test_accuracy:.2f}",
                "I don't know": random_result.dont_know,
                "Wrong confident": random_result.wrong_confident,
                "Stored paths": random_result.observer_paths,
            },
            {
                "Mode": guided_result.mode,
                "Training examples": guided_result.training_examples,
                "Test accuracy": f"{guided_result.test_accuracy:.2f}",
                "I don't know": guided_result.dont_know,
                "Wrong confident": guided_result.wrong_confident,
                "Stored paths": guided_result.observer_paths,
            },
        ],
        ["Mode", "Training examples", "Test accuracy", "I don't know", "Wrong confident", "Stored paths"],
    )


def main() -> None:
    set_seed(7)
    threshold = 0.92
    show_inference_examples(threshold)
    compare_selection_modes(threshold)


if __name__ == "__main__":
    main()
