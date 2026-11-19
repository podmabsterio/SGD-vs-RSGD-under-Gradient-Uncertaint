import sys
from pathlib import Path

import numpy as np


def finite_difference(function, point, step):
    gradient = np.empty_like(point)
    for coordinate in range(len(point)):
        offset = np.zeros_like(point)
        offset[coordinate] = step
        gradient[coordinate] = (
            function(point + offset) - function(point - offset)
        ) / (2 * step)
    return gradient


def main():
    from src.functions import TwoLayerNetwork

    function = TwoLayerNetwork(
        train_size=16, validation_size=8, test_size=8, data_seed=123
    )
    seeds = range(41, 46)
    indices = (0, 3)
    step = 1e-6
    relative_tolerance = 1e-5
    absolute_tolerance = 1e-8
    mean_errors, per_example_errors = [], []

    def check(observed, expected, errors):
        errors.append(float(np.max(np.abs(observed - expected))))
        np.testing.assert_allclose(
            observed, expected, rtol=relative_tolerance, atol=absolute_tolerance
        )

    for seed in seeds:
        point = function.initial_point(seed) + np.linspace(
            -0.1, 0.2, function.dimension
        )
        check(
            function.gradient(point),
            finite_difference(function.value, point, step),
            mean_errors,
        )
        for index in indices:

            def loss(parameters):
                logit = function.predict_logits(
                    parameters, function.x_train[index : index + 1]
                )[0]
                return np.logaddexp(0, logit) - function.y_train[index] * logit

            check(
                function.per_example_gradients(point, [index])[0],
                finite_difference(loss, point, step),
                per_example_errors,
            )

    print(
        f"Network gradient check: {function.dimension} coordinates, data seed 123, "
        "initialization seeds 41--45, examples 0 and 3"
    )
    print(
        f"Central differences: step={step:.0e}, "
        f"rtol={relative_tolerance:.0e}, atol={absolute_tolerance:.0e}"
    )
    print(f"Mean training gradient, max absolute error: {max(mean_errors):.3e}")
    print(f"Per-example gradients, max absolute error: {max(per_example_errors):.3e}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
