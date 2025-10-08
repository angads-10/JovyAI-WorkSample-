from typing import Any, Tuple
import numpy as np


def fitted_q_evaluation(dataset: Any, target_policy: Any) -> Tuple[float, Tuple[float, float]]:
    """Simple FQE with Monte Carlo value as proxy and bootstrapped CI.

    We approximate value by average return across trajectories as a baseline
    when no learned Q is available, returning bootstrap CIs.
    """
    if not dataset:
        return 0.0, (0.0, 0.0)
    traj_returns = np.array([float(np.sum(traj.get("rewards", []))) for traj in dataset], dtype=np.float64)
    est = float(traj_returns.mean())
    rng = np.random.default_rng(0)
    B = min(500, max(100, len(traj_returns) * 50))
    samples = []
    for _ in range(B):
        idx = rng.integers(0, len(traj_returns), size=len(traj_returns))
        samples.append(float(traj_returns[idx].mean()))
    ci = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
    return est, ci


