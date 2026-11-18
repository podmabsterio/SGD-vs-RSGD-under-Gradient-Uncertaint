import numpy as np

from .solvers import _info


class SGD:
    def __init__(
        self,
        learning_rate=0.1,
        decay=50.0,
        steps=500,
        name="sgd",
        convergence_tolerance=1e-3,
        patience=20,
    ):
        self.learning_rate = float(learning_rate)
        self.decay = float(decay)
        self.steps = steps
        self.name = name
        self.convergence_tolerance = float(convergence_tolerance)
        self.patience = patience
        self.reset()

    def reset(self):
        self.iteration = 0
        self.last_learning_rate = self.learning_rate
        self.last_info = _info()
        self.converged = False
        self.convergence_iteration = None
        self.stable_steps = 0

    def _update_convergence(self, point, displacement):
        if self.converged:
            return
        if not self.last_info["converged"]:
            self.stable_steps = 0
            return
        threshold = self.convergence_tolerance * (1 + np.linalg.norm(point))
        if np.linalg.norm(displacement) <= threshold:
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        if self.stable_steps >= self.patience:
            self.converged = True
            self.convergence_iteration = self.iteration

    @property
    def current_learning_rate(self):
        if self.decay == 0:
            return self.learning_rate
        return self.learning_rate / np.sqrt(1 + self.iteration / self.decay)

    def step(self, point, estimate):
        point = np.asarray(point, dtype=float)
        gradient = np.asarray(estimate["mean"], dtype=float)
        self.last_learning_rate = self.current_learning_rate
        displacement = -self.last_learning_rate * gradient
        self.last_info = _info(zero_step=not np.any(displacement))
        self.iteration += 1
        self._update_convergence(point, displacement)
        return point + displacement
