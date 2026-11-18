from abc import ABC, abstractmethod

import numpy as np


def gradient_estimate(observations):
    observations = np.asarray(observations, dtype=float)
    mean = observations.mean(axis=0)
    batch_size = len(observations)
    variance = np.square(observations - mean).sum(axis=0)
    variance /= batch_size * (batch_size - 1)
    epsilon = 1e-3 * max(float(variance.mean()), 1e-8)
    return {
        "mean": mean,
        "diagonal": variance + epsilon,
        "observations": observations,
        "epsilon": epsilon,
    }


class BaseFunction(ABC):
    @abstractmethod
    def value(self, point, split="train"):
        pass

    @abstractmethod
    def gradient(self, point):
        pass

    @abstractmethod
    def per_example_gradients(self, point, indices=None):
        pass

    @abstractmethod
    def initial_point(self, seed):
        pass

    @abstractmethod
    def accuracy(self, point, split="test"):
        pass

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def estimate_gradient(self, point):
        indices = self.rng.integers(self.train_size, size=self.batch_size)
        gradients = self.per_example_gradients(point, indices)
        noise = self.rng.standard_normal(gradients.shape)
        return gradient_estimate(gradients + self.noise_std * noise)

    def _point(self, point):
        return np.asarray(point, dtype=float)

    def _configure_observations(self, batch_size, noise_std, seed):
        self.batch_size = int(batch_size)
        self.noise_std = float(noise_std)
        self.reset(seed)


class BinaryClassification(BaseFunction):
    def __init__(
        self,
        train_size=512,
        validation_size=256,
        test_size=512,
        data_seed=2026,
        standardize=False,
        batch_size=64,
        noise_std=0.1,
    ):
        sizes = (train_size, validation_size, test_size)
        self.train_size, self.validation_size, self.test_size = sizes
        self.data_seed = data_seed
        self.standardize = standardize
        self.noise_variance = np.ones(self.dimension)
        sequences = np.random.SeedSequence(data_seed).spawn(3)
        for split, size, sequence in zip(
            ("train", "validation", "test"), sizes, sequences
        ):
            rng = np.random.default_rng(sequence)
            targets = np.repeat([0.0, 1.0], size // 2)
            features = self._generate_features(targets, rng)
            order = rng.permutation(size)
            setattr(self, f"x_{split}", features[order])
            setattr(self, f"y_{split}", targets[order])
        self.feature_mean = self.x_train.mean(axis=0) if standardize else np.zeros(2)
        self.feature_scale = self.x_train.std(axis=0) if standardize else np.ones(2)
        self.feature_scale = np.maximum(self.feature_scale, 1e-12)
        for split in ("train", "validation", "test"):
            features = getattr(self, f"x_{split}")
            setattr(
                self, f"x_{split}", (features - self.feature_mean) / self.feature_scale
            )
        self._configure_observations(batch_size, noise_std, data_seed)

    @abstractmethod
    def _generate_features(self, targets, rng):
        pass

    @abstractmethod
    def predict_logits(self, point, features):
        pass

    def _data(self, split):
        return getattr(self, f"x_{split}"), getattr(self, f"y_{split}")

    def value(self, point, split="train"):
        features, targets = self._data(split)
        logits = self.predict_logits(point, features)
        return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))

    def gradient(self, point):
        return self.per_example_gradients(point).mean(axis=0)

    def accuracy(self, point, split="test"):
        features, targets = self._data(split)
        return float(np.mean((self.predict_logits(point, features) >= 0) == targets))

    def _training_data(self, indices):
        if indices is None:
            return self.x_train, self.y_train
        return self.x_train[indices], self.y_train[indices]

    def _features(self, features):
        return np.asarray(features, dtype=float)
