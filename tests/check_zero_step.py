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
    cases = []
    for _ in range(30):
        square_root = rng.integers(1, 5) * np.array([1.0, 2.0])
        coordinate = rng.integers(2)
        gradient = np.zeros(2)
        gradient[coordinate] = rng.choice([-1, 1]) * square_root[coordinate]
        cases.append((gradient, square_root**2, rng.uniform(0.05, 0.25)))

    print("Zero-step threshold is rho = 1 in every case")
    for rho in (0.9, 1.0, 1.1):
        for name, solve in solvers.items():
            norms = []
            for gradient, diagonal, alpha in cases:
                step, info = solve(gradient, diagonal, alpha, rho)
                assert info["converged"], (name, info)
                norms.append(np.linalg.norm(step))
            values = np.asarray(norms)
            zeros = np.count_nonzero(values == 0.0)
            print(
                f"{name}, rho={rho}: {zeros}/{len(values)} zero steps, "
                f"step norm {values.mean():.3e} +/- {values.std(ddof=1):.3e}"
            )
            assert zeros == (len(cases) if rho >= 1 else 0), (name, rho)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
