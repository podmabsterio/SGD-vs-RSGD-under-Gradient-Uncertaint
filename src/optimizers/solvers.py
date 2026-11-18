import numpy as np
from scipy.optimize import minimize_scalar


def _info(iterations=0, converged=True, residual=0.0, zero_step=False):
    return {
        "iterations": iterations,
        "converged": converged,
        "residual": residual,
        "zero_step": zero_step,
    }


def _special_case(gradient, diagonal, alpha, rho):
    if rho == 0:
        displacement = -alpha * gradient
        return displacement, _info(zero_step=not np.any(displacement))
    if np.linalg.norm(gradient / np.sqrt(diagonal)) <= rho:
        return np.zeros_like(gradient), _info(zero_step=True)
    return None


def optimality_residual(gradient, diagonal, displacement, alpha, rho):
    gradient = np.asarray(gradient, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if rho == 0:
        residual = gradient + displacement / alpha
    else:
        radius = np.linalg.norm(np.sqrt(diagonal) * displacement)
        if radius == 0:
            return (
                0.0
                if np.linalg.norm(gradient / np.sqrt(diagonal)) <= rho
                else float("inf")
            )
        residual = (
            gradient + rho * diagonal * displacement / radius + displacement / alpha
        )
    residual_norm = np.linalg.norm(residual)
    gradient_norm = np.linalg.norm(gradient)
    if gradient_norm == 0:
        return 0.0 if residual_norm == 0 else float("inf")
    return float(residual_norm / gradient_norm)


def solve_brent(gradient, diagonal, alpha, rho, tolerance=1e-6, max_iterations=20000):
    gradient = np.asarray(gradient, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float)
    special = _special_case(gradient, diagonal, alpha, rho)
    if special is not None:
        return special
    upper = np.linalg.norm(np.sqrt(diagonal) * gradient)
    squared_gradient = np.square(gradient)
    shifted_diagonal = rho * diagonal

    def objective(fraction):
        radius = upper * fraction
        return (
            0.5
            * radius
            * (rho - np.sum(squared_gradient / (radius + shifted_diagonal)))
        )

    evaluations = 0
    for xatol in (tolerance, min(1e-12, tolerance * tolerance)):
        result = minimize_scalar(
            objective,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": xatol, "maxiter": max_iterations - evaluations},
        )
        evaluations += int(result.nfev)
        radius = upper * result.x
        displacement = -alpha * radius * gradient / (radius + shifted_diagonal)
        residual = optimality_residual(gradient, diagonal, displacement, alpha, rho)
        converged = bool(result.success and residual <= tolerance)
        if converged or max_iterations - evaluations < 2:
            break
    return displacement, _info(
        iterations=evaluations,
        converged=converged,
        residual=residual,
        zero_step=not np.any(displacement),
    )


def solve_proximal(
    gradient,
    diagonal,
    alpha,
    rho,
    tolerance=1e-6,
    max_iterations=20000,
    acceleration=False,
):
    gradient = np.asarray(gradient, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float)
    special = _special_case(gradient, diagonal, alpha, rho)
    if special is not None:
        return special
    square_root = np.sqrt(diagonal)
    tau = alpha * float(diagonal.min())
    linear = gradient / square_root
    curvature = 1 / (alpha * diagonal)
    z = np.zeros_like(gradient)
    extrapolated = z.copy()
    momentum = 1.0
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        value = extrapolated if acceleration else z
        candidate = value - tau * (linear + curvature * value)
        norm = np.linalg.norm(candidate)
        next_z = (
            np.zeros_like(candidate)
            if norm <= tau * rho
            else (1 - tau * rho / norm) * candidate
        )
        displacement = next_z / square_root
        residual = optimality_residual(gradient, diagonal, displacement, alpha, rho)
        if residual <= tolerance:
            return displacement, _info(
                iteration, True, residual, not np.any(displacement)
            )
        if acceleration:
            next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
            extrapolated = next_z + (momentum - 1) / next_momentum * (next_z - z)
            momentum = next_momentum
        z = next_z
    return displacement, _info(
        max_iterations, False, residual, not np.any(displacement)
    )
