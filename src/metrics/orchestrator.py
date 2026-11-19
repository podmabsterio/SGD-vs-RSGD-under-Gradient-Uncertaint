import json
from pathlib import Path

import numpy as np
import pandas as pd


class MetricOrchestrator:
    def __init__(self, metrics, save_budget_metrics=False):
        self.save_budget_metrics = save_budget_metrics
        self.metrics = {metric.name: metric for metric in metrics}
        self.names = list(self.metrics)
        self.iteration_names = [
            name for name, metric in self.metrics.items() if metric.save_per_iteration
        ]

    def evaluate(self, per_iteration=False, cached=None, **kwargs):
        names = self.iteration_names if per_iteration else self.names
        cached = {} if cached is None else cached
        return {
            name: float(
                cached[name] if name in cached else self.metrics[name](**kwargs)
            )
            for name in names
        }

    def aggregate(self, results):
        optimizers = list(dict.fromkeys(result["optimizer"] for result in results))
        last_record = max(
            (record["iteration"] for result in results for record in result["history"]),
            default=-1,
        )
        horizon = max([last_record] + [result.get("steps", -1) for result in results])
        columns = pd.Index(range(horizon + 1), name="iteration")
        index = pd.Index(optimizers, name="optimizer")
        tables = {}
        for name in self.iteration_names:
            mean = pd.DataFrame(index=index, columns=columns, dtype=float)
            std = pd.DataFrame(index=index, columns=columns, dtype=float)
            for optimizer in optimizers:
                histories = [
                    {
                        record["iteration"]: record.get(name, np.nan)
                        for record in result["history"]
                    }
                    for result in results
                    if result["optimizer"] == optimizer
                    and result["status"] == "success"
                ]
                values = pd.DataFrame(histories, columns=columns, dtype=float)
                values = values.replace([np.inf, -np.inf], np.nan)
                mean.loc[optimizer] = values.mean(axis=0)
                std.loc[optimizer] = values.std(axis=0, ddof=1)
            tables[name] = mean, std
        return tables

    def final_summary(self, results, snapshot_key="metrics", converged_only=False):
        names = self.names + ["runtime", "iterations"]
        summary = {}
        for optimizer in dict.fromkeys(result["optimizer"] for result in results):
            runs = [result for result in results if result["optimizer"] == optimizer]
            snapshots = [
                result[snapshot_key]
                for result in runs
                if result[snapshot_key] is not None
                and (not converged_only or result["converged"])
            ]
            metrics = {}
            for name in names:
                values = np.asarray(
                    [snapshot.get(name, np.nan) for snapshot in snapshots], dtype=float
                )
                values = values[np.isfinite(values)]
                with np.errstate(over="ignore", invalid="ignore"):
                    mean = float(values.mean()) if len(values) else None
                    std = float(values.std(ddof=1)) if len(values) > 1 else None
                metrics[name] = {
                    "mean": self._finite_or_none(mean),
                    "std": self._finite_or_none(std),
                }
            summary[optimizer] = {
                "metrics": metrics,
                "repetitions": len(runs),
                "converged_runs": sum(bool(result["converged"]) for result in runs),
                "budget_stops": sum(
                    result["status"] == "success" and not result["converged"]
                    for result in runs
                ),
                "failed_runs": sum(result["status"] == "failed" for result in runs),
                "seeds": [int(result["seed"]) for result in runs],
                "failures": [
                    {
                        "repeat": int(result["repeat"]),
                        "seed": int(result["seed"]),
                        "convergence_iteration": result.get("convergence_iteration"),
                        "error": result["error"],
                    }
                    for result in runs
                    if result["status"] == "failed"
                ],
            }
        return summary

    def save(self, results, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in self.names:
            if name not in self.iteration_names:
                for suffix in ("mean", "std"):
                    (output_dir / f"{name}_{suffix}.csv").unlink(missing_ok=True)
        tables = self.aggregate(results)
        for name, (mean, std) in tables.items():
            mean.to_csv(output_dir / f"{name}_mean.csv")
            std.to_csv(output_dir / f"{name}_std.csv")
        with (output_dir / "final_metrics.json").open("w") as stream:
            json.dump(self.final_summary(results), stream, indent=2, allow_nan=False)
            stream.write("\n")
        if self.save_budget_metrics:
            for filename, summary in (
                ("budget_metrics.json", self.final_summary(results, "budget_metrics")),
                (
                    "convergence_metrics.json",
                    self.final_summary(results, converged_only=True),
                ),
            ):
                with (output_dir / filename).open("w") as stream:
                    json.dump(summary, stream, indent=2, allow_nan=False)
                    stream.write("\n")
        return tables

    @staticmethod
    def _finite_or_none(value):
        return float(value) if value is not None and np.isfinite(value) else None
