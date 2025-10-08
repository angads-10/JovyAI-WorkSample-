import numpy as np

from data.dt_collator import collate_dt
from data.generate_toy import generate_toy_dataset
from ope.wis import weighted_importance_sampling
from ope.fqe import fitted_q_evaluation
from ope.dr import doubly_robust


def test_dt_collator_shapes():
    dataset = generate_toy_dataset(n_traj=3, T=5, obs_dim=4, act_dim=2, seed=0)
    batch = collate_dt(dataset, context_len=4)
    assert batch["rtg"].shape == (3, 4)
    assert batch["states"].shape == (3, 4, 4)
    assert batch["actions"].shape == (3, 4, 2)


def test_wis_runs_and_ci():
    dataset = generate_toy_dataset(n_traj=5, T=3, obs_dim=2, act_dim=1, seed=1)
    est, ci = weighted_importance_sampling(dataset, None, None)
    assert isinstance(est, float)
    assert isinstance(ci, tuple) and len(ci) == 2


def test_fqe_and_dr_stubs():
    dataset = generate_toy_dataset(n_traj=2, T=2)
    fqe_est, fqe_ci = fitted_q_evaluation(dataset, None)
    dr_est, dr_ci = doubly_robust(dataset, None, None)
    assert fqe_ci[0] <= fqe_est <= fqe_ci[1]
    assert dr_ci[0] <= dr_est <= dr_ci[1]


