import argparse
import itertools
import os
import time
from typing import Any, Dict, List

import yaml

from dashboards.metrics import MetricsLogger
from experiments.utils import set_deterministic_seeds, flatten_dict


def load_experiment_matrix(config_path: str) -> List[Dict[str, Any]]:
    """Load a YAML experiment matrix and expand into a list of runs.

    The YAML can be either a single config dict or a dict with a top-level
    key "matrix" containing a dict of parameter lists to grid over. Additional
    fixed keys under "base" are merged into each grid point.

    Example:
    base:
      seed: 42
      algo: iql
    matrix:
      lr: [1e-3, 3e-4]
      batch_size: [256, 512]
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if isinstance(cfg, list):
        return cfg

    base = cfg.get("base", {})
    matrix = cfg.get("matrix")
    if not matrix:
        return [base if isinstance(base, dict) else cfg]

    # Cartesian product over sorted keys for determinism
    keys = sorted(matrix.keys())
    values_product = itertools.product(*(matrix[k] for k in keys))
    runs: List[Dict[str, Any]] = []
    for combo in values_product:
        run = dict(base)
        run.update({k: v for k, v in zip(keys, combo)})
        runs.append(run)
    return runs


def get_run_name(run_cfg: Dict[str, Any]) -> str:
    flat = flatten_dict(run_cfg)
    parts = [f"{k}={flat[k]}" for k in sorted(flat.keys())]
    return ",".join(parts)


def train_one(run_cfg: Dict[str, Any], metrics: MetricsLogger) -> Dict[str, Any]:
    """Train one run given a fully-specified config.

    This function is a scaffold; algorithm-specific trainers will be plugged in
    via the "algo" field. It seeds deterministically, constructs the trainer,
    and logs key metrics. It returns a dict of final KPIs for aggregation.
    """
    seed = int(run_cfg.get("seed", 42))
    set_deterministic_seeds(seed)

    algo = str(run_cfg.get("algo", "iql")).lower()

    # Deferred imports (trainers will be implemented later)
    if algo == "iql":
        from algos.value_learning.iql import IQLTrainer  # type: ignore
        trainer_cls = IQLTrainer
    elif algo == "cql":
        from algos.value_learning.cql import CQLTrainer  # type: ignore
        trainer_cls = CQLTrainer
    elif algo == "dt":
        from algos.sequence_models.decision_transformer import DecisionTransformerTrainer  # type: ignore
        trainer_cls = DecisionTransformerTrainer
    else:
        raise ValueError(f"Unknown algo: {algo}")

    # Minimal trainer init surface; trainers should accept **run_cfg
    trainer = trainer_cls(**run_cfg)

    total_steps = int(run_cfg.get("train_steps", 1000))
    log_interval = int(run_cfg.get("log_interval", 100))

    start_time = time.time()
    last_log = start_time
    trainer.reset_metrics()
    for step in range(1, total_steps + 1):
        trainer.train_step()
        if step % log_interval == 0 or step == total_steps:
            kpis = trainer.get_metrics()
            kpis["step"] = step
            kpis["wall_time_s"] = time.time() - start_time
            metrics.log(kpis)
            last_log = time.time()

    final_kpis = trainer.get_metrics()
    final_kpis["wall_time_s"] = time.time() - start_time
    return final_kpis


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config or matrix")
    parser.add_argument("--outdir", type=str, default="runs", help="Directory for metrics CSVs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    runs = load_experiment_matrix(args.config)
    for idx, run_cfg in enumerate(runs):
        run_name = get_run_name(run_cfg)
        csv_path = os.path.join(args.outdir, f"{idx:03d}_{run_name}.csv")
        metrics = MetricsLogger(csv_path)
        metrics.print_header(f"Run {idx+1}/{len(runs)}: {run_name}")
        final = train_one(run_cfg, metrics)
        metrics.print_summary(final)


if __name__ == "__main__":
    main()


