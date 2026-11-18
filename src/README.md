# Experiments

Use Python 3.12 and run commands from the project root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
python run.py
python run.py --config-name quadratic_solver_comparison
```

The config contains one `function`, a list of `optimizers`, a list of
`metrics`, and execution settings. Objects are created directly with Hydra
`instantiate`. Batch size and observation noise belong to the function.
Learning rate, schedule, fixed update budget and convergence settings belong
to each optimizer. All optimizers must have the same `steps` value.
The progress bar tracks completed optimizer-repeat runs.

```bash
python run.py function.noise_std=0 function.batch_size=128
python run.py metrics.0.save_per_iteration=true
```

Every repeat executes the entire fixed budget. At the first convergence
signal, its current metrics, elapsed runtime and iteration number are copied
to a final snapshot. Later updates continue to produce table values and never
change that snapshot. For example, snapshots at iterations 350 and 400 with
a budget of 500 produce a final mean iteration count of 375. Their tables
still contain every iteration through 500.

The convergence signal measures relative parameter stability: an update
must satisfy `norm(step) <= convergence_tolerance * (1 + norm(point))` for
`patience` consecutive iterations. The signal and its iteration remain fixed
until the optimizer is reset. These parameters are distinct from the inner
R-SGD solver's optimality tolerance. If a repeat reaches the budget without
signalling convergence, its last metrics, full runtime and budget iteration
number become its final snapshot. Reaching the budget does not set the
convergence flag. JSON records converged runs separately from budget stops.

Each experiment directory contains:

- `<metric_name>_mean.csv` and `<metric_name>_std.csv` for metrics with
  `save_per_iteration: true`. Rows are optimizer names, columns are iterations
  starting at zero. Each cell aggregates that iteration across successful
  repeats. By default, all metrics are final-only and these CSVs are absent.
- `final_metrics.json`: per-optimizer mean and sample standard deviation for
  every metric, including `iterations` and `runtime`. Each repeat contributes
  its first-convergence snapshot or, if it never converged, its budget-end
  snapshot. The file separately records converged runs, budget stops and
  failures.
- `config.yaml`: the composed configuration, including overrides and seeds.

Parameter points, raw histories, datasets and graphs are not saved. Sample
standard deviation uses `ddof=1`, so it is undefined with only one available
repeat. Unsupported or unavailable values are empty in CSV and `null` in
JSON. A failure after convergence does not overwrite an existing snapshot;
its full-budget history is excluded from the iteration tables.

Metric classes inherit from `BaseMetric`, which provides
`save_per_iteration=False`. Each class has a unique snake_case `name` and a
callable interface. Every configured metric is included in the final JSON,
regardless of this flag. A final-only metric is evaluated only when its final
snapshot is captured. An iteration metric is evaluated at the initial point
and after each update; its value is reused when that update becomes the final
snapshot. Final-only values are not stored in the iteration history.

```yaml
metrics:
  - _target_: src.metrics.TrainLoss
    save_per_iteration: true
  - _target_: src.metrics.TestAccuracy
    save_per_iteration: false
```

Each metric receives the function, optimizer, current and previous points,
gradient estimate, iteration counters and seeds as keyword arguments.
Unused arguments are accepted with `**kwargs`. Runtime is measured by the
runner rather than a metric class and appears only in the final JSON.

The `quadratic_solver_comparison` config compares 12 R-SGD instances on
the quadratic from Section 2. Brent and proximal both use tolerances
`1e-4`, `1e-2`, `1e-1`, and `3e-1`. Proximal runs with and without
acceleration. Each setting has 100 repeats of 500 updates. Results go to
`output/quadratic_solver_comparison`. All four selected metrics are saved
per iteration. Validation loss uses the existing metric name `validation_loss`.
`acceleration=true` enables FISTA extrapolation for
the proximal method. Both inner solvers divide the optimality residual by
the gradient norm before comparing it with the tolerance. `solver_iterations`
counts function evaluations for Brent and iterations for proximal gradient.

Configuration values are used directly. Runtime failures are recorded when
parameters or the objective become nonfinite, or an inner solver fails to
reach its required accuracy.

Datasets have a separate fixed seed. Every repeat uses separate derived
seeds for initialization and sampling, matched across optimizers. Thus
variability across repeats measures training randomness on one dataset.
Sampling is with replacement, and noise affects gradient observations only.
The covariance of the batch mean and the positive diagonal correction
follow the assignment. Both models use stable, unregularized binary
cross-entropy. The two-layer network has eight tanh hidden units and 33
parameters, with Xavier-initialized weights and zero biases.

```bash
python -m pytest -p no:capture src/tests -q
```
