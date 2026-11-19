import sys
from pathlib import Path

import numpy as np


def main():
    from src.functions import NoisyQuadratic

    repeats = 1000
    print(f"Batch-size check at the initial point, {repeats} batches per size")
    print("b    trace(S), mean +/- std    b * mean trace(S)    b * empirical trace")
    for batch_size in (8, 32, 128, 512):
        function = NoisyQuadratic(batch_size=batch_size, noise_std=1.0)
        function.reset(42 + batch_size)
        point = function.initial_point(42)
        estimates = [function.estimate_gradient(point) for _ in range(repeats)]
        diagonal = np.array([estimate["diagonal"] for estimate in estimates])
        gradients = np.array([estimate["mean"] for estimate in estimates])
        trace = diagonal.sum(axis=1)
        empirical = gradients.var(axis=0, ddof=1)
        expected = function.noise_std**2 * function.noise_variance

        print(
            f"{batch_size:>3}  {trace.mean():.5e} +/- {trace.std(ddof=1):.5e}"
            f"    {batch_size * trace.mean():.4f}"
            f"    {batch_size * empirical.sum():.4f}"
        )
        np.testing.assert_allclose(
            batch_size * diagonal.mean(axis=0), expected, rtol=0.08
        )
        np.testing.assert_allclose(batch_size * empirical, expected, rtol=0.15)
    print(f"Theoretical scaled trace without diagonal correction: {expected.sum():.4f}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
