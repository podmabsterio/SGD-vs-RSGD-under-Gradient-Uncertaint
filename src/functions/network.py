import numpy as np
from scipy.special import expit

from src.functions.base import BinaryClassification


class TwoLayerNetwork(BinaryClassification):
    def __init__(
        self,
        train_size=512,
        validation_size=256,
        test_size=512,
        data_seed=2026,
        standardize=False,
        hidden_size=8,
        batch_size=64,
        noise_std=0.1,
    ):
        self.hidden_size = hidden_size
        self.dimension = 4 * hidden_size + 1
        super().__init__(
            train_size,
            validation_size,
            test_size,
            data_seed,
            standardize,
            batch_size,
            noise_std,
        )

    def _generate_features(self, targets, rng):
        angle = rng.uniform(0, np.pi, len(targets))
        first = targets == 0
        features = np.column_stack(
            (
                np.where(first, np.cos(angle), 1 - np.cos(angle)),
                np.where(first, np.sin(angle), 0.5 - np.sin(angle)),
            )
        )
        return features + rng.normal(0, 0.1, features.shape)

    def initial_point(self, seed):
        rng = np.random.default_rng(seed)
        first_limit = np.sqrt(6 / (2 + self.hidden_size))
        second_limit = np.sqrt(6 / (self.hidden_size + 1))
        first_weights = rng.uniform(-first_limit, first_limit, (self.hidden_size, 2))
        second_weights = rng.uniform(-second_limit, second_limit, self.hidden_size)
        return np.concatenate(
            (first_weights.ravel(), np.zeros(self.hidden_size), second_weights, [0.0])
        )

    def _parameters(self, point):
        point = self._point(point)
        hidden = self.hidden_size
        return (
            point[: 2 * hidden].reshape(hidden, 2),
            point[2 * hidden : 3 * hidden],
            point[3 * hidden : 4 * hidden],
            point[-1],
        )

    def predict_logits(self, point, features):
        first_weights, first_bias, second_weights, second_bias = self._parameters(point)
        features = self._features(features)
        hidden = np.tanh(features @ first_weights.T + first_bias)
        return hidden @ second_weights + second_bias

    def per_example_gradients(self, point, indices=None):
        features, targets = self._training_data(indices)
        first_weights, first_bias, second_weights, second_bias = self._parameters(point)
        hidden = np.tanh(features @ first_weights.T + first_bias)
        residual = expit(hidden @ second_weights + second_bias) - targets
        hidden_gradient = residual[:, None] * second_weights * (1 - hidden**2)
        first_gradient = hidden_gradient[:, :, None] * features[:, None, :]
        return np.concatenate(
            (
                first_gradient.reshape(len(features), -1),
                hidden_gradient,
                residual[:, None] * hidden,
                residual[:, None],
            ),
            axis=1,
        )
