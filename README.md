JovyAI Offline RL Starter (Compact)

Quick start
- Install: python>=3.10, optional torch. pip install -r requirements.txt (or your env)
- Run training: python -m experiments.train --config experiments/configs/experiments.yaml --outdir runs
- Run OPE eval: python -m experiments.eval_ope --config experiments/configs/ope_example.yaml --out ope_eval.csv

What’s included
- IQL/CQL/DT trainers (to be filled), experiment matrix runner, CSV/console metrics
- Compact helpers in-place (seeding, replay buffer) to minimize files
- OPE scaffold (WIS, FQE, DR) interfaces; robust impls to plug in
- Safety hooks for action masking; deterministic toy MARL env scaffold

Config matrix (YAML)
Use base + matrix keys. Example in `experiments/configs/experiments.yaml`.

Determinism
- Global seeds set per run. Disable cuDNN autotune for repeatability.

Structure
- experiments/train.py: CLI, matrix, seeding, replay buffer, trainer loop
- dashboards/metrics.py: CSV + console logger
- experiments/eval_ope.py: OPE CLI (WIS/FQE/DR)

Notes & References
- See `docs/survey_notes.md` for brief citations (IQL, CQL, DT, OPE).

