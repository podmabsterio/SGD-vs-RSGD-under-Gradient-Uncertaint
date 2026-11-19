import random
import time

import numpy as np


def _check_finite(function, point):
    if not np.isfinite(point).all() or not np.isfinite(function.value(point)):
        raise ArithmeticError("Nonfinite parameters or objective")


def run_single(function, optimizer, collector, repeat, seed):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    initial_seed, sampling_seed = [
        int(sequence.generate_state(1)[0])
        for sequence in np.random.SeedSequence(seed).spawn(2)
    ]
    function.reset(sampling_seed)
    optimizer.reset()
    point = function.initial_point(initial_seed)
    state = {
        "point": point,
        "previous_point": point.copy(),
        "estimate": None,
        "iteration": 0,
        "repeat": repeat,
        "seed": seed,
        "gradient_evaluations": 0,
        "zero_steps": 0,
        "step_norm_sum": 0.0,
        "relative_step_sum": 0.0,
        "solver_iterations": 0,
        "max_solver_residual": 0.0,
    }
    history = []
    values = {}
    final_metrics = None
    budget_metrics = None
    save_budget_metrics = getattr(collector, "save_budget_metrics", False)
    converged = False
    status, error = "success", ""
    try:
        _check_finite(function, point)
        values = collector.evaluate(
            per_iteration=True, function=function, optimizer=optimizer, **state
        )
        if values:
            history.append({"iteration": state["iteration"], **values})
        for iteration in range(1, optimizer.steps + 1):
            estimate = function.estimate_gradient(point)
            next_point = optimizer.step(point, estimate)
            step_norm = float(np.linalg.norm(next_point - point))
            denominator = optimizer.last_learning_rate * np.linalg.norm(
                estimate["mean"]
            )
            state.update(
                previous_point=point,
                point=next_point,
                estimate=estimate,
                iteration=iteration,
                gradient_evaluations=state["gradient_evaluations"]
                + function.batch_size,
                zero_steps=state["zero_steps"] + int(optimizer.last_info["zero_step"]),
                step_norm_sum=state["step_norm_sum"] + step_norm,
                relative_step_sum=state["relative_step_sum"]
                + (step_norm / denominator if denominator > 0 else 0.0),
                solver_iterations=state["solver_iterations"]
                + optimizer.last_info["iterations"],
                max_solver_residual=max(
                    state["max_solver_residual"], optimizer.last_info["residual"]
                ),
            )
            point = next_point
            _check_finite(function, point)
            values = collector.evaluate(
                per_iteration=True, function=function, optimizer=optimizer, **state
            )
            if values:
                history.append({"iteration": state["iteration"], **values})
            if not optimizer.last_info["converged"]:
                raise RuntimeError(
                    f"Inner solver did not converge at update {iteration}: "
                    f"residual={optimizer.last_info['residual']:.3g}"
                )
            if optimizer.converged and final_metrics is None:
                final_metrics = {
                    **collector.evaluate(
                        cached=values, function=function, optimizer=optimizer, **state
                    ),
                    "iterations": iteration,
                    "runtime": time.perf_counter() - started,
                }
                converged = True
        if final_metrics is None or save_budget_metrics:
            if (
                final_metrics is not None
                and final_metrics["iterations"] == state["iteration"]
            ):
                budget_metrics = final_metrics
            else:
                budget_metrics = {
                    **collector.evaluate(
                        cached=values, function=function, optimizer=optimizer, **state
                    ),
                    "iterations": state["iteration"],
                }
            if final_metrics is None:
                final_metrics = budget_metrics
    except Exception as exception:
        status = "failed"
        error = f"{type(exception).__name__}: {exception}"
    total_runtime = time.perf_counter() - started
    convergence_iteration = final_metrics["iterations"] if converged else None
    if final_metrics is not None and not converged:
        final_metrics["runtime"] = total_runtime
    result = {
        "function": type(function).__name__,
        "optimizer": optimizer.name,
        "repeat": repeat,
        "seed": seed,
        "initial_seed": initial_seed,
        "sampling_seed": sampling_seed,
        "noise_std": function.noise_std,
        "steps": optimizer.steps,
        "iterations": state["iteration"],
        "converged": converged,
        "convergence_iteration": convergence_iteration,
        "total_runtime": total_runtime,
        "status": status,
        "error": error,
        "metrics": final_metrics,
        "history": history,
    }
    if save_budget_metrics:
        result["budget_metrics"] = (
            {**budget_metrics, "runtime": total_runtime}
            if budget_metrics is not None
            else None
        )
    return result
