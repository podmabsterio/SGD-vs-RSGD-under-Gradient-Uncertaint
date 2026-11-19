import sys
from pathlib import Path
from statistics import fmean


def main():
    from src.experiment import run_single
    from src.functions import NoisyQuadratic
    from src.metrics import BaseMetric, MetricOrchestrator, TrainLoss
    from src.optimizers import RobustSGD, SGD

    class Coordinate(BaseMetric):
        def __init__(self, coordinate):
            super().__init__(save_per_iteration=True)
            self.coordinate = coordinate
            self.name = f"theta_{coordinate + 1}"

        def __call__(self, point, **kwargs):
            return point[self.coordinate]

    steps = 50
    repeats = 20
    function = NoisyQuadratic(noise_std=1.0)
    collector = MetricOrchestrator(
        [TrainLoss(save_per_iteration=True), Coordinate(0), Coordinate(1)]
    )

    def run_history(optimizer, repeat):
        result = run_single(function, optimizer, collector, repeat, seed=42 + repeat)
        assert result["status"] == "success", result["error"]
        assert len(result["history"]) == steps + 1
        return result["history"]

    sgd = SGD(steps=steps)
    baseline = [run_history(sgd, repeat) for repeat in range(repeats)]
    baseline_loss = fmean(history[-1]["train_loss"] for history in baseline)
    print(f"Mean SGD loss at step {steps}: {baseline_loss:.6e}")

    for rho in (0.0, 1e-3):
        robust = RobustSGD(steps=steps, rho=rho, tolerance=1e-6)
        label = f"R-SGD (rho={rho:g}) minus SGD"
        pairs = []
        for repeat, plain in enumerate(baseline):
            history = [
                {
                    "iteration": sgd_step["iteration"],
                    **{
                        name: robust_step[name] - sgd_step[name]
                        for name in collector.names
                    },
                }
                for sgd_step, robust_step in zip(
                    plain, run_history(robust, repeat), strict=True
                )
            ]
            pairs.append({"optimizer": label, "status": "success", "history": history})

        tables = collector.aggregate(pairs)
        print(f"{label} over {repeats} paired runs")
        for name, (mean, std) in tables.items():
            peak = max(
                abs(record[name]) for pair in pairs for record in pair["history"]
            )
            print(
                f"{name} at step {steps}: {mean.iat[0, -1]:+.3e} +/- "
                f"{std.iat[0, -1]:.3e}, max absolute difference over all steps: {peak:.3e}"
            )
            if rho == 0:
                assert peak == 0.0, f"{name} differs at rho=0"

        if rho > 0:
            output = Path("output/checks/zero_radius_check")
            output.mkdir(parents=True, exist_ok=True)
            mean, std = tables["train_loss"]
            mean.to_csv(output / "train_loss_difference_mean.csv")
            std.to_csv(output / "train_loss_difference_std.csv")
            print(
                f"Mean final loss difference relative to SGD: {mean.iat[0, -1] / baseline_loss:+.3%}"
            )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
