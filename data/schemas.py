from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


