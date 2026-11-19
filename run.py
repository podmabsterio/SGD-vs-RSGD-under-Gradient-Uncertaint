from pathlib import Path

import hydra
from hydra.utils import instantiate
from joblib import Parallel, delayed, parallel_config
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.experiment import run_single
from src.metrics import MetricOrchestrator


@hydra.main(
    version_base="1.3",
    config_path="src/configs",
    config_name="quadratic_solver_comparison",
)
def main(config: DictConfig):
    function = instantiate(config.function)
    optimizers = instantiate(config.optimizers)
    collector = MetricOrchestrator(
        instantiate(config.metrics),
        save_budget_metrics=config.get("save_budget_metrics", False),
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=output_dir / "config.yaml", resolve=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results_iter = Parallel(n_jobs=config.n_jobs, return_as="generator_unordered")(
            delayed(run_single)(
                function, optimizer, collector, repeat, config.seed + repeat
            )
            for optimizer in optimizers
            for repeat in range(config.repeats)
        )
        results = list(
            tqdm(
                results_iter,
                total=len(optimizers) * config.repeats,
                desc="Runs",
                unit="run",
            )
        )
    optimizer_order = {
        optimizer.name: index for index, optimizer in enumerate(optimizers)
    }
    results.sort(
        key=lambda result: (optimizer_order[result["optimizer"]], result["repeat"])
    )
    collector.save(results, output_dir)
    print(f"Results saved to {output_dir}")
    failures = [result for result in results if result["status"] != "success"]
    if failures:
        print(f"Failed runs: {len(failures)}. See final_metrics.json for details.")


if __name__ == "__main__":
    main()
