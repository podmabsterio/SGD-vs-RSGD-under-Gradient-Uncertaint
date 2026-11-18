import argparse
from itertools import combinations, cycle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def overlapping_curves(ax, means):
    transform = ax.yaxis.get_transform()
    height = np.ptp(transform.transform(ax.get_ylim()))
    displayed = transform.transform(means.ravel()).reshape(means.shape)
    overlapping = set()
    for first, second in combinations(range(len(means)), 2):
        close = np.isclose(
            displayed[first], displayed[second], atol=0.01 * height, rtol=0
        )
        if close.mean() >= 0.8:
            overlapping.update((first, second))
    return overlapping


def style_overlaps(ax, overlapping, steps):
    stride = max(steps // 8, len(overlapping))
    styles = cycle(("--", "-.", ":"))
    markers = cycle(("o", "s", "^", "D", "v", "P", "X", "<", ">", "*"))
    for offset, index in enumerate(sorted(overlapping)):
        ax.lines[index].set(
            linestyle=next(styles),
            marker=next(markers),
            markevery=(offset * stride // len(overlapping), stride),
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=0.9,
            alpha=0.85,
        )
        ax.collections[index].set_alpha(0.08)


def plot_metric(mean_path, std_path, metric, output_path):
    mean = pd.read_csv(mean_path, index_col=0)
    std = pd.read_csv(std_path, index_col=0).reindex_like(mean)
    iterations = mean.columns.to_numpy(dtype=int)
    means = mean.to_numpy(dtype=float)

    figure = Figure(figsize=(8.6, 4.8), layout="constrained")
    ax = figure.subplots()
    for optimizer, values, spread in zip(mean.index, means, std.to_numpy(dtype=float)):
        (line,) = ax.plot(iterations, values, label=optimizer, linewidth=1.4)
        ax.fill_between(
            iterations,
            values - spread,
            values + spread,
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.grid(alpha=0.25)
    overlapping = overlapping_curves(ax, means)

    magnitudes = np.abs(means)
    nonzero = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
    linthresh = float(nonzero.min()) if nonzero.size else 1.0
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.autoscale_view()
    overlapping.update(overlapping_curves(ax, means))
    style_overlaps(ax, overlapping, len(iterations))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    figure.savefig(output_path.with_stem(f"{output_path.stem}_log"))

    ax.set_yscale("linear")
    ax.autoscale_view()
    figure.savefig(output_path)


def main():
    parser = argparse.ArgumentParser(description="Plot mean and standard deviation.")
    parser.add_argument("data", type=Path, help="Experiment directory")
    parser.add_argument("-o", "--output", type=Path, help="Directory for PDF plots")
    args = parser.parse_args()
    mean_paths = sorted(args.data.glob("*_mean.csv"))
    if not mean_paths:
        parser.error(f"No *_mean.csv files in {args.data}")

    output = args.output or Path("plots") / args.data.resolve().name
    output.mkdir(parents=True, exist_ok=True)
    for mean_path in mean_paths:
        metric = mean_path.name.removesuffix("_mean.csv")
        std_path = args.data / f"{metric}_std.csv"
        plot_metric(mean_path, std_path, metric, output / f"{metric}.pdf")


if __name__ == "__main__":
    main()
