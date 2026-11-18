import numpy as np

from .base import BaseMetric


class TrainLoss(BaseMetric):
    name = "train_loss"

    def __call__(self, function, point, **kwargs):
        return function.value(point, split="train")


class ValidationLoss(BaseMetric):
    name = "validation_loss"

    def __call__(self, function, point, **kwargs):
        return function.value(point, split="validation")


class TestLoss(BaseMetric):
    name = "test_loss"

    def __call__(self, function, point, **kwargs):
        return function.value(point, split="test")


class TestAccuracy(BaseMetric):
    name = "test_accuracy"

    def __call__(self, function, point, **kwargs):
        return function.accuracy(point, split="test")


class TrainAccuracy(BaseMetric):
    name = "train_accuracy"

    def __call__(self, function, point, **kwargs):
        return function.accuracy(point, split="train")


class GradientNorm(BaseMetric):
    name = "gradient_norm"

    def __call__(self, function, point, **kwargs):
        return np.linalg.norm(function.gradient(point))


class StepNorm(BaseMetric):
    name = "step_norm"

    def __call__(self, point, previous_point, **kwargs):
        return np.linalg.norm(point - previous_point)


class MeanStepNorm(BaseMetric):
    name = "mean_step_norm"

    def __call__(self, iteration, step_norm_sum, **kwargs):
        return step_norm_sum / max(iteration, 1)


class MeanRelativeStep(BaseMetric):
    name = "mean_relative_step"

    def __call__(self, iteration, relative_step_sum, **kwargs):
        return relative_step_sum / max(iteration, 1)


class GradientEvaluations(BaseMetric):
    name = "gradient_evaluations"

    def __call__(self, gradient_evaluations, **kwargs):
        return gradient_evaluations


class ZeroStepFrequency(BaseMetric):
    name = "zero_step_frequency"

    def __call__(self, iteration, zero_steps, **kwargs):
        return zero_steps / max(iteration, 1)


class SolverIterations(BaseMetric):
    name = "solver_iterations"

    def __call__(self, solver_iterations, **kwargs):
        return solver_iterations


class SolverResidual(BaseMetric):
    name = "solver_residual"

    def __call__(self, optimizer, **kwargs):
        return optimizer.last_info.get("residual", 0.0)


class MaxSolverResidual(BaseMetric):
    name = "max_solver_residual"

    def __call__(self, max_solver_residual, **kwargs):
        return max_solver_residual


class UncertaintyTrace(BaseMetric):
    name = "uncertainty_trace"

    def __call__(self, estimate, **kwargs):
        if estimate is None:
            return float("nan")
        return np.sum(estimate["diagonal"])
