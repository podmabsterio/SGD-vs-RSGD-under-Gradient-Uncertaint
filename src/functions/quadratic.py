import numpy as np

from src.functions.base import BaseFunction, gradient_estimate


class NoisyQuadratic(BaseFunction):
    def __init__(
        self,
        curvature=(1, 4),
        initial=(2, 2),
        noise_scales=(2, 0.5),
        batch_size=64,
        noise_std=0.1,
    ):
        self.curvature = np.asarray(curvature, dtype=float)
        self.initial = np.asarray(initial, dtype=float)
        self.noise_scales = np.asarray(noise_scales, dtype=float)
        self.dimension = len(self.curvature)
        self.noise_variance = self.noise_scales**2
        self.train_size = 1
        self.validation_size = 1
        self.test_size = 1
        self._configure_observations(batch_size, noise_std, 0)

    def value(self, point, split="train"):
        point = self._point(point)
        return float(0.5 * np.dot(self.curvature, point**2))

    def gradient(self, point):
        return self.curvature * self._point(point)

    def per_example_gradients(self, point, indices=None):
        size = 1 if indices is None else len(np.atleast_1d(indices))
        return np.tile(self.gradient(point), (size, 1))

    def estimate_gradient(self, point):
        noise = self.rng.standard_normal((self.batch_size, self.dimension))
        observations = self.gradient(point) + self.noise_std * self.noise_scales * noise
        return gradient_estimate(observations)

    def initial_point(self, seed):
        return self.initial.copy()

    def accuracy(self, point, split="test"):
        return float("nan")
