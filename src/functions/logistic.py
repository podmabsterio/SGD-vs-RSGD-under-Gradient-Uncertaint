import numpy as np
from scipy.special import expit

from src.functions.base import BinaryClassification


class LogisticRegression(BinaryClassification):
    dimension = 3

    def _generate_features(self, targets, rng):
        return (2 * targets[:, None] - 1) + rng.standard_normal((len(targets), 2))

    def initial_point(self, seed):
        return np.zeros(self.dimension)

    def predict_logits(self, point, features):
        point = self._point(point)
        features = self._features(features)
        return features @ point[:2] + point[2]

    def per_example_gradients(self, point, indices=None):
        features, targets = self._training_data(indices)
        residual = expit(self.predict_logits(point, features)) - targets
        return np.column_stack((residual[:, None] * features, residual))
