import argparse
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from src.experiment import run_single
from src.optimizers import SGD, RobustSGD


class Endpoint:
    point = None

    def evaluate(self, *, point, iteration, optimizer, **kwargs):
        if iteration == optimizer.steps:
            self.point = point
        return {}


def fit(config, function, seed):
    optimizers = instantiate(config.optimizers)
    for cls in (SGD, RobustSGD):
        optimizer = next(
            optimizer for optimizer in optimizers if type(optimizer) is cls
        )
        endpoint = Endpoint()
        result = run_single(function, optimizer, endpoint, seed - config.seed, seed)
        if result["status"] != "success":
            raise RuntimeError(f"{optimizer.name}, seed {seed}: {result['error']}")
        yield endpoint.point


def plot(configs, seed, output):
    functions = [instantiate(config.function) for config in configs]
    features = np.concatenate([function.x_test for function in functions])
    low, high = features.min(axis=0) - 0.2, features.max(axis=0) + 0.2
    x, y = np.meshgrid(*(np.linspace(a, b, 160) for a, b in zip(low, high)))
    grid = np.column_stack((x.ravel(), y.ravel()))
    cols = min(2, len(configs))
    rows = (len(configs) + cols - 1) // cols
    figure = Figure(figsize=(3.6 * cols, 3.3 * rows + 0.4), layout="constrained")
    axes = figure.subplots(rows, cols, squeeze=False, sharex=True, sharey=True).ravel()
    classes = ((0, "#0072B2", "o"), (1, "#D55E00", "^"))
    methods = (("SGD", "#292929", "-", 2.0), ("R-SGD", "#A61E5C", "--", 1.6))

    for ax, config, function in zip(axes, configs, functions):
        for target, color, marker in classes:
            points = function.x_test[function.y_test == target]
            ax.scatter(
                *points.T, s=10, color=color, marker=marker, alpha=0.27, linewidths=0
            )
        for point, (_, color, style, width) in zip(
            fit(config, function, seed), methods
        ):
            logits = function.predict_logits(point, grid).reshape(x.shape)
            ax.contour(
                x,
                y,
                logits,
                levels=[0],
                colors=[color],
                linestyles=[style],
                linewidths=[width],
            )
        ax.set_title(rf"$b={function.batch_size}$, $\sigma={function.noise_std:g}$")
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    for ax in axes[len(configs) :]:
        ax.set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            label=f"Test class {target}",
            color=color,
            marker=marker,
            markersize=5,
            linestyle="none",
            alpha=0.6,
        )
        for target, color, marker in classes
    ] + [
        Line2D([], [], label=name, color=color, linestyle=style, linewidth=width)
        for name, color, style, width in methods
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


def main():
    parser = argparse.ArgumentParser(
        description="Plot decision boundaries at a fixed evaluation seed"
    )
    parser.add_argument(
        "configs", nargs="+", type=Path, help="Saved evaluation configs"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Training seed to replay"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output PDF path"
    )
    args = parser.parse_args()
    configs = [OmegaConf.load(path) for path in args.configs]
    if any(
        not config.seed <= args.seed < config.seed + config.repeats
        for config in configs
    ):
        parser.error("The seed must be among the recorded repetitions of each config")
    plot(configs, args.seed, args.output)


if __name__ == "__main__":
    main()
