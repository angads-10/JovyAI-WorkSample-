from typing import Any, Tuple
import numpy as np


def weighted_importance_sampling(dataset: Any, behavior_policy: Any, target_policy: Any) -> Tuple[float, Tuple[float, float]]:
    """Simple WIS estimator stub with normalizing weights and basic CI via bootstrap.

    Expects dataset as list of trajectories, each trajectory a dict with fields
    'rewards' and 'logp_b', 'logp_t' (behavior and target log-probs per step).
    Returns (estimate, (ci_low, ci_high)).
    """
    if not dataset:
        return 0.0, (0.0, 0.0)

    returns = []
    weights = []
    for traj in dataset:
        r = float(np.sum(traj.get("rewards", [])))
        logp_b = np.array(traj.get("logp_b", []), dtype=np.float32)
        logp_t = np.array(traj.get("logp_t", []), dtype=np.float32)
        w = float(np.exp((logp_t - logp_b).sum()))
        returns.append(r)
        weights.append(w)

    weights = np.array(weights, dtype=np.float64)
    returns = np.array(returns, dtype=np.float64)
    if weights.sum() == 0:
        estimate = 0.0
    else:
        estimate = float(np.sum(weights * returns) / np.sum(weights))

    # Bootstrap CI
    rng = np.random.default_rng(0)
    B = min(200, max(50, len(returns) * 20))
    samples = []
    for _ in range(B):
        idx = rng.integers(0, len(returns), size=len(returns))
        w = weights[idx]
        r = returns[idx]
        val = float(np.sum(w * r) / (np.sum(w) + 1e-12))
        samples.append(val)
    ci = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
    return estimate, ci


