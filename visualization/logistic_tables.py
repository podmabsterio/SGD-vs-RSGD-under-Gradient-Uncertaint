import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def format_stat(run, metric, precision, scale=1):
    stats = run["metrics"][metric]
    mean, std = stats["mean"], stats["std"]
    if mean is None:
        return "--"
    value = f"{mean * scale:.{precision}f}"
    if std is not None:
        value += rf" \pm {std * scale:.{precision}f}"
    return "$" + value + "$"


def tabular(columns, headers, rows):
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{columns}}}",
            r"\hline",
            " & ".join(headers) + r" \\",
            r"\hline",
            *(" & ".join(row) + r" \\" for row in rows),
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )


def read_rows(run_dirs):
    budget_rows, convergence_rows = [], []
    for directory in run_dirs:
        config = OmegaConf.load(directory / "config.yaml")
        labels = {
            optimizer.name: "R-SGD"
            if optimizer._target_.endswith(".RobustSGD")
            else "SGD"
            for optimizer in config.optimizers
        }
        budget = json.loads((directory / "budget_metrics.json").read_text())
        convergence = json.loads((directory / "convergence_metrics.json").read_text())
        for name, endpoint in budget.items():
            hit = convergence[name]
            stop = hit if hit["converged_runs"] else endpoint
            condition = [
                str(config.function.batch_size),
                f"{config.function.noise_std:g}",
                labels[name],
            ]
            budget_rows.append(
                condition
                + [
                    format_stat(endpoint, "train_loss", 4),
                    format_stat(endpoint, "test_loss", 4),
                    format_stat(endpoint, "test_accuracy", 2, 100),
                    format_stat(endpoint, "runtime", 1, 1000),
                    format_stat(endpoint, "zero_step_frequency", 2, 100),
                    format_stat(endpoint, "gradient_evaluations", 0),
                ]
            )
            convergence_rows.append(
                condition
                + [
                    f"{hit['converged_runs']}/{hit['repetitions']}",
                    format_stat(stop, "iterations", 1),
                    format_stat(stop, "runtime", 1, 1000),
                    format_stat(stop, "test_accuracy", 2, 100),
                    str(hit["failed_runs"]),
                ]
            )
    return budget_rows, convergence_rows


def main():
    parser = argparse.ArgumentParser(
        description="Write LaTeX tables for logistic runs."
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Evaluation run directories")
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/logistic"))
    args = parser.parse_args()

    budget_rows, convergence_rows = read_rows(args.runs)
    prefix = [r"$b$", r"$\sigma$", "Method"]
    budget_header = prefix + [
        "Train loss",
        "Test loss",
        r"Acc. (\%)",
        "Time (ms)",
        r"Zero (\%)",
        "Grad. eval.",
    ]
    convergence_header = prefix + [
        "Hits",
        "Update",
        "Time (ms)",
        r"Acc. (\%)",
        "Failed",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "budget_table.tex").write_text(
        tabular("rrlrrrrrr", budget_header, budget_rows)
    )
    (args.output / "convergence_table.tex").write_text(
        tabular("rrlrrrrr", convergence_header, convergence_rows)
    )


if __name__ == "__main__":
    main()
