from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class HospitalDetMARL:
    """Deterministic toy multi-agent hospital environment (minimal stub)."""

    def __init__(self, num_agents: int = 2, horizon: int = 10, seed: int = 42) -> None:
        self.num_agents = num_agents
        self.horizon = horizon
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        return {f"agent_{i}": np.array([i, 0.0], dtype=np.float32) for i in range(self.num_agents)}

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        self.t += 1
        obs = {k: np.array([int(k.split("_")[1]), float(self.t)], dtype=np.float32) for k in actions}
        rews = {k: float(np.clip(a).mean()) for k, a in actions.items()}
        done = self.t >= self.horizon
        info: Dict = {}
        return obs, rews, done, info


