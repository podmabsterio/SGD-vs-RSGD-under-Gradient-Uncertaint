import sys
from functools import partial
from pathlib import Path

import numpy as np


def main():
    from src.optimizers import solve_brent, solve_proximal

    solvers = {
        "Brent": solve_brent,
        "Proximal": solve_proximal,
        "Accelerated proximal": partial(solve_proximal, acceleration=True),
    }
    tolerance = 1e-6
    rng = np.random.default_rng(42)
    residuals = {name: [] for name in solvers}
    for _ in range(30):
        gradient = rng.normal(size=5)
        diagonal = np.exp(rng.uniform(-2, 2, size=5))
        alpha = rng.uniform(0.05, 0.25)
        rho = 0.5 * np.linalg.norm(gradient / np.sqrt(diagonal))
        for name, solve in solvers.items():
            step, info = solve(gradient, diagonal, alpha, rho, tolerance=tolerance)
            assert info["converged"] and np.any(step), (name, info)
            weighted_step = diagonal * step
            radius = np.sqrt(step @ weighted_step)
            residual = gradient + rho * weighted_step / radius + step / alpha
            residuals[name].append(np.linalg.norm(residual) / np.linalg.norm(gradient))

    print(f"Relative optimality residual with tolerance {tolerance:.0e}")
    for name, samples in residuals.items():
        values = np.asarray(samples)
        print(
            f"{name}: {values.mean():.3e} +/- {values.std(ddof=1):.3e}, "
            f"max: {values.max():.3e}"
        )
        assert values.max() <= tolerance, name


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
