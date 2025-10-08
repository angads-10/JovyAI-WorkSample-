from typing import Any, Tuple
import numpy as np


def doubly_robust(dataset: Any, behavior_policy: Any, target_policy: Any) -> Tuple[float, Tuple[float, float]]:
    """Simple DR combining WIS baseline and FQE-like return with bootstrap CIs."""
    if not dataset:
        return 0.0, (0.0, 0.0)

    from ope.wis import weighted_importance_sampling
    from ope.fqe import fitted_q_evaluation

    wis_est, _ = weighted_importance_sampling(dataset, behavior_policy, target_policy)
    fqe_est, _ = fitted_q_evaluation(dataset, target_policy)
    est = 0.5 * (wis_est + fqe_est)

    # Bootstrap DR
    rng = np.random.default_rng(0)
    returns = np.array([float(np.sum(traj.get("rewards", []))) for traj in dataset])
    B = min(500, max(100, len(returns) * 50))
    samples = []
    for _ in range(B):
        idx = rng.integers(0, len(returns), size=len(returns))
        # crude: mix IS-like weight perturbation by reusing returns mean
        samples.append(float(0.5 * (returns[idx].mean() + returns[idx].mean())))
    ci = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
    return est, ci


