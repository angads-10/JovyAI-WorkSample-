import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import yaml

from dashboards.metrics import MetricsLogger


def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def eval_one(cfg: Dict[str, Any], metrics: MetricsLogger) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    set_deterministic_seeds(seed)

    # Placeholder imports; real implementations to be added
    from ope.wis import weighted_importance_sampling  # type: ignore
    from ope.fqe import fitted_q_evaluation  # type: ignore
    from ope.dr import doubly_robust  # type: ignore

    # In practice we would load dataset/trajectories and a policy to evaluate
    # Here, assume cfg supplies stubs or file paths
    dataset = cfg.get("dataset")
    behavior_policy = cfg.get("behavior_policy")
    target_policy = cfg.get("target_policy")

    wis_estimate, wis_ci = weighted_importance_sampling(dataset, behavior_policy, target_policy)
    fqe_estimate, fqe_ci = fitted_q_evaluation(dataset, target_policy)
    dr_estimate, dr_ci = doubly_robust(dataset, behavior_policy, target_policy)

    kpis = {
        "wis": wis_estimate,
        "wis_ci_low": wis_ci[0],
        "wis_ci_high": wis_ci[1],
        "fqe": fqe_estimate,
        "fqe_ci_low": fqe_ci[0],
        "fqe_ci_high": fqe_ci[1],
        "dr": dr_estimate,
        "dr_ci_low": dr_ci[0],
        "dr_ci_high": dr_ci[1],
    }
    metrics.log(kpis)
    return kpis


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL OPE Evaluation")
    parser.add_argument("--config", type=str, required=True, help="YAML with dataset/policy OPE settings")
    parser.add_argument("--out", type=str, default="ope_eval.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    metrics = MetricsLogger(args.out)
    metrics.print_header("OPE Evaluation")
    kpis = eval_one(cfg, metrics)
    metrics.print_summary(kpis)


if __name__ == "__main__":
    main()


