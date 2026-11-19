import argparse
import sys
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate
from matplotlib.figure import Figure
from omegaconf import OmegaConf


LOSS_LABELS = {"train_loss": "Training loss", "validation_loss": "Validation loss"}


def budget_statistics(data, metric, optimizers):
    mean = pd.read_csv(data / f"{metric}_mean.csv", index_col=0)
    std = pd.read_csv(data / f"{metric}_std.csv", index_col=0)
    return [
        (
            spec,
            mean.at[spec.name, str(spec.steps)],
            std.at[spec.name, str(spec.steps)],
        )
        for spec in optimizers
    ]


def plot_sweep(data, optimizers, output):
    decays = sorted({spec.decay for spec in optimizers})
    rates = sorted({spec.learning_rate for spec in optimizers})
    for metric, ylabel in LOSS_LABELS.items():
        statistics = budget_statistics(data, metric, optimizers)
        figure = Figure(figsize=(4.6, 3.0), layout="constrained")
        ax = figure.subplots()
        for decay in decays:
            rows = sorted(
                (spec.learning_rate, mean, std)
                for spec, mean, std in statistics
                if spec.decay == decay
            )
            rate, mean, std = zip(*rows, strict=True)
            label = "Constant" if decay == 0 else f"Decay = {decay:g}"
            ax.errorbar(rate, mean, yerr=std, marker="o", capsize=3, label=label)
        ax.set_xscale("log")
        ax.set_xticks(rates, [f"{rate:g}" for rate in rates])
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.set(xlabel=r"Initial learning rate $\alpha_0$", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        figure.savefig(output / f"sgd_{metric}_vs_lr.pdf")


def plot_schedule(optimizers, output, label_methods=True):
    schedules = {}
    for spec in optimizers:
        optimizer = instantiate(spec)
        method = "R-SGD" if spec._target_.endswith(".RobustSGD") else "SGD"
        key = optimizer.learning_rate, optimizer.decay, optimizer.steps
        schedules.setdefault(key, (optimizer, set()))[1].add(method)

    figure = Figure(figsize=(6.3, 3.6), layout="constrained")
    ax = figure.subplots()
    for (rate, decay, steps), (optimizer, methods) in schedules.items():
        learning_rates = []
        for k in range(steps):
            optimizer.iteration = k
            learning_rates.append(optimizer.current_learning_rate)
        label = rf"$\alpha_0={rate:g}$, decay = {decay:g}"
        if label_methods:
            label = f"{' and '.join(sorted(methods, reverse=True))}: {label}"
        ax.plot(range(steps), learning_rates, label=label)
    ax.set(xlabel=r"Update index $k$", ylabel=r"Learning rate $\alpha_k$")
    ax.grid(alpha=0.25)
    ax.legend()
    figure.savefig(output / "learning_rate_schedule.pdf")


def write_grid(data, optimizers, output):
    if len({spec.decay for spec in optimizers}) != 1:
        raise ValueError("The R-SGD grid requires one shared decay setting.")
    rates = sorted({spec.learning_rate for spec in optimizers})
    radii = sorted({spec.get("rho", 0) for spec in optimizers})
    row_end = r" \\"
    for metric in LOSS_LABELS:
        rows = budget_statistics(data, metric, optimizers)
        values = {
            (spec.learning_rate, spec.get("rho", 0)): mean for spec, mean, _ in rows
        }
        best = min(mean for mean in values.values() if pd.notna(mean))
        headers = [r"$\alpha_0$"] + [rf"$\rho={rho:g}$" for rho in radii]
        lines = [
            r"\begin{tabular}{r" + "r" * len(radii) + "}",
            r"\hline",
            " & ".join(headers) + row_end,
            r"\hline",
        ]
        for rate in rates:
            cells = [f"${rate:g}$"]
            for rho in radii:
                mean = values.get((rate, rho), float("nan"))
                if pd.isna(mean):
                    cells.append("--")
                elif mean == best:
                    cells.append(rf"$\mathbf{{{mean:.5f}}}$")
                else:
                    cells.append(f"${mean:.5f}$")
            lines.append(" & ".join(cells) + row_end)
        lines.extend([r"\hline", r"\end{tabular}"])
        (output / f"rsgd_{metric}_grid.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description="Format optimizer tuning results.")
    parser.add_argument("mode", choices=("sweep", "schedule", "grid"))
    parser.add_argument("data", type=Path, help="Experiment directory")
    parser.add_argument("-o", "--output", type=Path, help="Directory for PDFs or TeX")
    parser.add_argument("--names", nargs="+", help="Optimizer names for schedule mode")
    args = parser.parse_args()
    optimizers = OmegaConf.load(args.data / "config.yaml").optimizers
    if args.names:
        if args.mode != "schedule":
            parser.error("--names is only supported in schedule mode.")
        by_name = {spec.name: spec for spec in optimizers}
        unknown = [name for name in args.names if name not in by_name]
        if unknown:
            parser.error(f"Unknown optimizer names: {', '.join(unknown)}")
        optimizers = [by_name[name] for name in args.names]
    output = args.output or Path("plots") / args.data.resolve().name
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "sweep":
        plot_sweep(args.data, optimizers, output)
    elif args.mode == "schedule":
        plot_schedule(optimizers, output, label_methods=args.names is None)
    else:
        write_grid(args.data, optimizers, output)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
