import numpy as np

from .sgd import SGD
from .solvers import solve_brent, solve_proximal


class RobustSGD(SGD):
    def __init__(
        self,
        learning_rate=0.1,
        decay=50.0,
        rho=1.0,
        solver="brent",
        tolerance=1e-6,
        max_iterations=20000,
        acceleration=False,
        steps=500,
        name="rsgd",
        convergence_tolerance=1e-3,
        patience=20,
    ):
        super().__init__(
            learning_rate=learning_rate,
            decay=decay,
            steps=steps,
            name=name,
            convergence_tolerance=convergence_tolerance,
            patience=patience,
        )
        self.rho = float(rho)
        self.solver = solver
        self.tolerance = float(tolerance)
        self.max_iterations = max_iterations
        self.acceleration = bool(acceleration)

    def step(self, point, estimate):
        point = np.asarray(point, dtype=float)
        gradient = np.asarray(estimate["mean"], dtype=float)
        diagonal = np.asarray(estimate["diagonal"], dtype=float)
        self.last_learning_rate = self.current_learning_rate
        arguments = (
            gradient,
            diagonal,
            self.last_learning_rate,
            self.rho,
            self.tolerance,
            self.max_iterations,
        )
        if self.solver == "brent":
            displacement, self.last_info = solve_brent(*arguments)
        else:
            displacement, self.last_info = solve_proximal(
                *arguments, acceleration=self.acceleration
            )
        self.iteration += 1
        self._update_convergence(point, displacement)
        return point + displacement
