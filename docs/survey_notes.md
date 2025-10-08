Notes and references (concise)

- IQL: Expectile regression for V values; advantage-weighted policy learning.
  - "Implicit Q-Learning for Offline RL" (Kostrikov et al., 2021).
- CQL: Adds conservative log-sum-exp penalty to push down Q on OOD actions.
  - "Conservative Q-Learning for Offline RL" (Kumar et al., 2020), arXiv:2006.04779.
- Decision Transformer: Sequence modeling of (RTG, state, action) with causal masking.
  - "Decision Transformer: Reinforcement Learning via Sequence Modeling" (Chen et al., 2021).
- OPE:
  - WIS: Normalize trajectory IS weights to reduce variance.
  - FQE: Fit Q and evaluate target policy; bootstrap for CIs.
  - DR: Combine IS with model-based correction for bias-variance trade-off.
  - "A Doubly Robust OPE Estimator for Episodic RL" (Jiang & Li, 2016).
  - "OPE via FQE" (Le et al., 2019).
- Safety / support constraints: Reject actions outside empirical support; action masking.
  - Support constraints and uncertainty-aware filtering commonly used in safe offline RL.


