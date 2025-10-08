from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class DecisionTransformerTrainer:
    """Minimal DT with teacher forcing (rtg, state, prev action) and causal mask.

    Uses a tiny Transformer when torch is available; otherwise, stub loss.
    """

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg
        self.step = 0
        self.metrics: Dict[str, float] = {}
        self.rng = np.random.default_rng(int(cfg.get("seed", 42)))
        self.context_len = int(cfg.get("context_len", 20))
        self.lr = float(cfg.get("lr", 3e-4))

        self.dataset = cfg.get("dataset") or self._generate_toy_dataset()
        self.obs_dim = self.dataset[0]["states"].shape[-1]
        self.act_dim = self.dataset[0]["actions"].shape[-1]

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            d_model = 64
            self.torch = torch
            self.nn = nn
            self.embed_r = nn.Linear(1, d_model)
            self.embed_s = nn.Linear(self.obs_dim, d_model)
            self.embed_a = nn.Linear(self.act_dim, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.head = nn.Linear(d_model, self.act_dim)
            self.opt = optim.Adam(self.parameters(), lr=self.lr)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for m in [self.embed_r, self.embed_s, self.embed_a, self.transformer, self.head]:
                m.to(self.device)
            self._use_torch = True
        except Exception:
            self._use_torch = False

    def parameters(self):
        if not self._use_torch:
            return []
        for m in [self.embed_r, self.embed_s, self.embed_a, self.transformer, self.head]:
            for p in m.parameters():
                yield p

    def _generate_toy_dataset(self) -> List[Dict[str, np.ndarray]]:
        from data.generate_toy import generate_toy_dataset

        return generate_toy_dataset(n_traj=10, T=30, obs_dim=int(self.cfg.get("obs_dim", 8)), act_dim=int(self.cfg.get("act_dim", 2)), seed=int(self.cfg.get("seed", 42)))

    def reset_metrics(self) -> None:
        self.metrics = {"loss": 0.0}

    def _collate(self, batch_traj: List[Dict]) -> Dict[str, np.ndarray]:
        from data.dt_collator import collate_dt

        return collate_dt(batch_traj, self.context_len)

    def train_step(self) -> None:
        self.step += 1
        if not self._use_torch:
            loss = float(self.rng.random())
            self.metrics["loss"] = 0.9 * self.metrics.get("loss", loss) + 0.1 * loss
            return

        import torch

        idx = self.rng.integers(0, len(self.dataset), size=16)
        batch_traj = [self.dataset[i] for i in idx]
        collated = self._collate(batch_traj)
        rtg = torch.as_tensor(collated["rtg"], device=self.device).unsqueeze(-1).float()
        states = torch.as_tensor(collated["states"], device=self.device).float()
        prev_actions = torch.as_tensor(collated["actions"], device=self.device).float()

        # Teacher forcing: predict current action given (rtg, state, prev action)
        er = self.embed_r(rtg)
        es = self.embed_s(states)
        ea = self.embed_a(prev_actions)
        seq = torch.stack([er, es, ea], dim=2)  # [B, T, 3, D]
        B, T, K, D = seq.shape
        seq = seq.view(B, T * K, D)

        # Causal mask so each token attends to previous tokens only
        mask = torch.triu(torch.ones(T * K, T * K, device=self.device), diagonal=1) == 1
        mask = mask.float() * -1e9
        z = self.transformer(seq, mask=mask)
        # Take last token positions corresponding to action slots
        action_positions = torch.arange(2, T * K, 3, device=self.device)
        z_a = z[:, action_positions, :]
        pred = self.head(z_a)
        target = torch.as_tensor([d["actions"][:T] for d in batch_traj], device=self.device).float()

        loss = ((pred - target) ** 2).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.metrics["loss"] = 0.9 * self.metrics.get("loss", float(loss.detach().cpu().item())) + 0.1 * float(loss.detach().cpu().item())

    def get_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)


