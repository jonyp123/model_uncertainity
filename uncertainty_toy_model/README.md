# Uncertainty Toy Model

This is a small research prototype for uncertainty-guided learning in a capacity-limited neural network.

The model sees three numeric inputs and predicts one numeric output. It is intentionally tiny: one hidden layer with six neurons. The point is not high performance. The point is to observe when a small network follows a familiar internal activation path and when it should say "I don't know".

## Patterns

The generated patterns are:

- Pattern A: `[x, x+1, x+2] -> x+3`
- Pattern B: `[x, 2x, 3x] -> 4x`
- Pattern C: `[x, x^2, x^3] -> x^4`
- Pattern D: `[x, x+2, x+4] -> x+6`

The initial model trains only on Pattern A and Pattern B. It is then tested on all four patterns.

## Activation-Path Similarity

During training, the prototype records hidden-layer activations for examples the model predicts correctly. These activation vectors are grouped by pattern name.

At inference time, the model:

1. Runs the input through the network.
2. Extracts the hidden activation vector.
3. Compares that vector with stored successful activation paths using cosine similarity.
4. Allows a prediction only if the best similarity crosses a threshold.
5. Otherwise returns "I don't know".

This makes uncertainty visible as a mismatch between the current internal path and known successful paths.

## Learning Modes

The experiment compares two simple modes:

- Random example selection: when uncertain, add random examples from all available patterns.
- Uncertainty-guided selection: when uncertain on a pattern, add more examples from that same uncertain pattern.

The printed comparison includes:

- number of training examples used
- test accuracy on predictions that were allowed
- number of "I don't know" responses
- number of wrong confident predictions
- number of stored activation paths

## Run

```bash
pip install -r requirements.txt
python main.py
```

The output includes individual inference examples and a small comparison table.
