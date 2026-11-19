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
    rng = np.random.default_rng(42)
    errors = {name: [] for name in solvers}
    for _ in range(30):
        gradient = rng.normal(size=2)
        scale = rng.uniform(0.5, 4.0)
        alpha = rng.uniform(0.05, 0.25)
        shrinkage = rng.uniform(0.2, 0.8)
        diagonal = np.full(2, scale)
        rho = shrinkage * np.linalg.norm(gradient) / np.sqrt(scale)
        expected = -alpha * (1 - shrinkage) * gradient
        for name, solve in solvers.items():
            step, info = solve(gradient, diagonal, alpha, rho)
            assert info["converged"], (name, info)
            errors[name].append(step - expected)

    print("Signed coordinate error: computed step minus isotropic formula")
    for name, differences in errors.items():
        values = np.asarray(differences).ravel()
        print(
            f"{name}: {values.mean():+.3e} +/- {values.std(ddof=1):.3e}, "
            f"max absolute error: {np.abs(values).max():.3e}"
        )
        np.testing.assert_allclose(values, 0.0, atol=1e-6, rtol=0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
