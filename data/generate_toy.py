from typing import List, Dict
import numpy as np


def generate_toy_dataset(n_traj: int = 10, T: int = 20, obs_dim: int = 4, act_dim: int = 2, seed: int = 0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_traj):
        states = rng.standard_normal((T, obs_dim)).astype(np.float32)
        actions = rng.standard_normal((T, act_dim)).astype(np.float32)
        rewards = (states.mean(axis=1) + actions.mean(axis=1)).astype(np.float32)
        logp_b = rng.standard_normal((T,)).astype(np.float32)
        logp_t = logp_b + 0.1 * rng.standard_normal((T,)).astype(np.float32)
        data.append({"states": states, "actions": actions, "rewards": rewards, "logp_b": logp_b, "logp_t": logp_t})
    return data


