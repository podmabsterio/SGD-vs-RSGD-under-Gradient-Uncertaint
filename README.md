# SGD versus Robust SGD under Gradient Uncertainty

This project studies how uncertainty in mini-batch gradients affects plain
SGD and robust SGD. It checks the robust step on a noisy quadratic and
compares the two methods on logistic regression and a small two-layer network.

## Install and run

From the repository root, using Python 3.11:

```bash
python -m pip install -r requirements.txt
python run.py --config-name logistic_evaluation output_dir=output/example
```

The example runs both optimizers on logistic regression and saves the resolved
configuration, per-iteration CSV summaries, and final JSON metrics in
`output/example/`.

## Repository structure

- `run.py` runs repeated experiments and saves their results.
- `src/configs/` contains the Hydra YAML configurations. Each defines one
  objective, a list of optimizers and metrics, and the run settings.
- `src/functions/`, `src/optimizers/`, and `src/metrics/` contain the
  objectives, update rules and inner solvers, and metric collection.
- `tests/` contains standalone numerical checks.
- `visualization/` builds plots and LaTeX tables from saved results.
- `output/` contains saved experiment results.
- `plots/` contains figures and LaTeX tables for the report.

## Report and slides

The [report](report.pdf) presents the derivations, numerical checks, and
experimental results. The [presentation](presentation.pdf) summarizes the
main theoretical results and the comparison of SGD and robust SGD.

### Reproducing the report

**Choosing the inner solver.** The Brent and proximal gradient comparison
uses [`quadratic_solver_comparison.yaml`](src/configs/quadratic_solver_comparison.yaml).

**Numerical checks.** Run the scripts from the repository root, for example
with `python tests/check_zero_radius.py`.

| Report check | Script |
| --- | --- |
| Zero and small radius | [`check_zero_radius.py`](tests/check_zero_radius.py) |
| Isotropic step formula | [`check_isotropic.py`](tests/check_isotropic.py) |
| Zero-step threshold | [`check_zero_step.py`](tests/check_zero_step.py) |
| Optimality of nonzero steps | [`check_optimality.py`](tests/check_optimality.py) |
| Effect of batch size on gradient uncertainty | [`check_batch_size.py`](tests/check_batch_size.py) |
| Two-layer network gradients | [`check_network_gradient.py`](tests/check_network_gradient.py) |

**Simulation 1 and Simulation 2.** Pass the configuration name without
`.yaml` to `--config-name`.

| Part | Simulation 1 (logistic regression) | Simulation 2 (two-layer network) |
| --- | --- | --- |
| Data and shared settings | [`logistic_base.yaml`](src/configs/logistic_base.yaml) | [`network_base.yaml`](src/configs/network_base.yaml) |
| SGD learning rate and schedule | [`logistic_sgd_tuning.yaml`](src/configs/logistic_sgd_tuning.yaml) | [`network_sgd_tuning.yaml`](src/configs/network_sgd_tuning.yaml) |
| R-SGD learning rate and radius grid | [`logistic_rsgd_tuning.yaml`](src/configs/logistic_rsgd_tuning.yaml) | [`network_rsgd_tuning.yaml`](src/configs/network_rsgd_tuning.yaml) |
| Radius sensitivity with the schedule held fixed | [`logistic_radius_sensitivity.yaml`](src/configs/logistic_radius_sensitivity.yaml) | [`network_radius_sensitivity.yaml`](src/configs/network_radius_sensitivity.yaml) |
| Evaluation with selected settings | [`logistic_evaluation.yaml`](src/configs/logistic_evaluation.yaml) | [`network_evaluation.yaml`](src/configs/network_evaluation.yaml) |

The tuning and evaluation configurations inherit the corresponding base
file. For evaluation, vary `function.batch_size` and `function.noise_std`
and set a separate `output_dir` for each condition. The batch sweep uses
sizes 16, 64, and 256 with zero added noise. The noise sweep uses batch size
64 and standard deviations 0, 0.25, and 0.5 for logistic regression or
0, 0.1, and 0.2 for the network. Both simulations also pair batch size 16
with their highest noise level.
