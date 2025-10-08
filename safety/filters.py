from typing import Optional, Tuple
import numpy as np


def apply_action_mask(actions: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return actions
    if mask.shape != actions.shape:
        return actions
    return np.where(mask > 0.5, actions, 0.0)


def compute_action_support(actions: np.ndarray, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    low = actions.min(axis=0) - margin
    high = actions.max(axis=0) + margin
    return low, high


def reject_ood_actions(actions: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.where((actions >= low) & (actions <= high), actions, 0.0)


