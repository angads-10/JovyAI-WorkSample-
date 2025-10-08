from typing import Dict, List
import numpy as np


def collate_dt(trajectories: List[Dict], context_len: int) -> Dict[str, np.ndarray]:
    # Minimal collator for DT: (rtg, state, prev_action)
    batch = {"rtg": [], "states": [], "actions": []}
    for traj in trajectories:
        rtg = np.cumsum(traj["rewards"][::-1])[::-1]
        T = min(len(rtg), context_len)
        batch["rtg"].append(rtg[:T])
        batch["states"].append(traj["states"][:T])
        prev_actions = np.vstack([np.zeros_like(traj["actions"][0:1]), traj["actions"][:-1]])
        batch["actions"].append(prev_actions[:T])
    for k in batch:
        batch[k] = np.stack(batch[k], axis=0)
    return batch


