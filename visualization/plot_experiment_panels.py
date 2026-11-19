import argparse
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("noise", "batch"))
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--start-update", type=int, default=0)
    args = parser.parse_args()
    configs = [OmegaConf.load(run / "config.yaml") for run in args.runs]
    gradient_budget = min(
        config.function.batch_size * config.optimizers[0].steps for config in configs
    )
    figure = Figure(figsize=(10.4, 5.1), layout="constrained")
    axes = figure.subplots(2, len(args.runs), sharey="row", squeeze=False)
    for column, (run, config) in enumerate(zip(args.runs, configs, strict=True)):
        size = config.function.batch_size
        sigma = config.function.noise_std
        metrics = (
            ("train_loss", "validation_loss")
            if args.mode == "noise"
            else ("train_loss",) * 2
        )
        for row, metric in enumerate(metrics):
            ax = axes[row, column]
            mean = pd.read_csv(run / f"{metric}_mean.csv", index_col=0)
            std = pd.read_csv(run / f"{metric}_std.csv", index_col=0).reindex_like(mean)
            columns = mean.columns[
                mean.columns.to_numpy(dtype=int) >= args.start_update
            ]
            x = columns.to_numpy(dtype=int)
            examples = args.mode == "batch" and row == 1
            if examples:
                x = x * size
                ax.set_xlim(0, gradient_budget)
            for optimizer, style in zip(config.optimizers, ("-", "--"), strict=True):
                values = mean.loc[optimizer.name, columns].to_numpy(dtype=float)
                spread = std.loc[optimizer.name, columns].to_numpy(dtype=float)
                (line,) = ax.plot(x, values, style, label=optimizer.name, linewidth=1.4)
                ax.fill_between(
                    x,
                    values - spread,
                    values + spread,
                    color=line.get_color(),
                    alpha=0.16,
                )
            ax.set_xlabel("Example gradients" if examples else "Updates")
            ax.grid(alpha=0.25)
            if column == 0:
                ax.set_ylabel(metric.replace("_", " ").capitalize())
            if row == 0:
                ax.set_title(f"$b={size}$, $\\sigma={sigma:g}$")
    axes[0, 0].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
