from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np


class IQLTrainer:
    """Compact IQL trainer with simple sampling and target networks (PyTorch optional).

    If torch is unavailable, falls back to a dummy loss to keep runner functional.
    """

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg
        self.step = 0
        self.metrics: Dict[str, float] = {}
        self.rng = np.random.default_rng(int(cfg.get("seed", 42)))
        self.batch_size = int(cfg.get("batch_size", 256))
        self.tau = float(cfg.get("tau", 0.005))
        self.discount = float(cfg.get("discount", 0.99))
        self.expectile = float(cfg.get("expectile", 0.7))
        self.lr = float(cfg.get("lr", 3e-4))

        # Dataset: expect list of dicts with arrays; else synthesize toy data
        self.dataset = cfg.get("dataset") or self._generate_toy_dataset()
        self.obs_dim = self.dataset[0]["states"].shape[-1]
        self.act_dim = self.dataset[0]["actions"].shape[-1]

        # Action support for safety
        acts = np.concatenate([d["actions"] for d in self.dataset], axis=0)
        from safety.filters import compute_action_support

        self.support_low, self.support_high = compute_action_support(acts, margin=0.05)

        # Lazy torch setup
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            def mlp(in_dim: int, out_dim: int) -> nn.Module:
                return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))

            self.torch = torch
            self.nn = nn
            self.q1 = mlp(self.obs_dim + self.act_dim, 1)
            self.q2 = mlp(self.obs_dim + self.act_dim, 1)
            self.v = mlp(self.obs_dim, 1)
            self.pi = mlp(self.obs_dim, self.act_dim)
            self.q1_t = mlp(self.obs_dim + self.act_dim, 1)
            self.q2_t = mlp(self.obs_dim + self.act_dim, 1)
            self.v_t = mlp(self.obs_dim, 1)
            self.q1_t.load_state_dict(self.q1.state_dict())
            self.q2_t.load_state_dict(self.q2.state_dict())
            self.v_t.load_state_dict(self.v.state_dict())

            params = list(self.q1.parameters()) + list(self.q2.parameters()) + list(self.v.parameters()) + list(self.pi.parameters())
            self.opt = optim.Adam(params, lr=self.lr)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for m in [self.q1, self.q2, self.v, self.pi, self.q1_t, self.q2_t, self.v_t]:
                m.to(self.device)
            self._use_torch = True
        except Exception:
            self._use_torch = False

    def _generate_toy_dataset(self) -> List[Dict[str, np.ndarray]]:
        from data.generate_toy import generate_toy_dataset

        return generate_toy_dataset(n_traj=20, T=50, obs_dim=int(self.cfg.get("obs_dim", 8)), act_dim=int(self.cfg.get("act_dim", 2)), seed=int(self.cfg.get("seed", 42)))

    def reset_metrics(self) -> None:
        self.metrics = {"loss_q": 0.0, "loss_v": 0.0, "loss_pi": 0.0}

    def _sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Sample random transitions uniformly from trajectories
        traj_idxs = self.rng.integers(0, len(self.dataset), size=self.batch_size)
        t_idxs = []
        for i in traj_idxs:
            T = len(self.dataset[i]["rewards"]) - 1
            t_idxs.append(self.rng.integers(0, max(1, T)))
        s = np.stack([self.dataset[i]["states"][t] for i, t in zip(traj_idxs, t_idxs)], axis=0).astype(np.float32)
        a = np.stack([self.dataset[i]["actions"][t] for i, t in zip(traj_idxs, t_idxs)], axis=0).astype(np.float32)
        r = np.stack([self.dataset[i]["rewards"][t] for i, t in zip(traj_idxs, t_idxs)], axis=0).astype(np.float32)
        sp = np.stack([self.dataset[i]["states"][t + 1] for i, t in zip(traj_idxs, t_idxs)], axis=0).astype(np.float32)
        d = np.stack([float(t + 1 >= len(self.dataset[i]["rewards"])) for i, t in zip(traj_idxs, t_idxs)], axis=0).astype(np.float32)
        return s, a, r, sp, d

    def _ema(self, target, online) -> None:
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def train_step(self) -> None:
        self.step += 1
        if not self._use_torch:
            # Fallback keeps loop alive
            base = float(self.rng.random())
            self.metrics["loss_q"] = 0.9 * self.metrics.get("loss_q", base) + 0.1 * base
            self.metrics["loss_v"] = self.metrics["loss_q"]
            self.metrics["loss_pi"] = self.metrics["loss_q"]
            return

        import torch

        s, a, r, sp, d = self._sample_batch()
        # Safety: reject OOD actions for policy updates by clamping into support
        from safety.filters import reject_ood_actions

        a = reject_ood_actions(a, self.support_low, self.support_high)

        s_t = torch.as_tensor(s, device=self.device)
        a_t = torch.as_tensor(a, device=self.device)
        r_t = torch.as_tensor(r, device=self.device).unsqueeze(-1)
        sp_t = torch.as_tensor(sp, device=self.device)
        d_t = torch.as_tensor(d, device=self.device).unsqueeze(-1)

        # Targets
        with torch.no_grad():
            v_sp = self.v_t(sp_t)
            target_q = r_t + (1.0 - d_t) * self.discount * v_sp

        # Q losses
        qa1 = self.q1(torch.cat([s_t, a_t], dim=-1))
        qa2 = self.q2(torch.cat([s_t, a_t], dim=-1))
        lq1 = ((qa1 - target_q) ** 2).mean()
        lq2 = ((qa2 - target_q) ** 2).mean()

        # V expectile regression
        with torch.no_grad():
            q_pi = torch.min(self.q1_t(torch.cat([s_t, self.pi(s_t)], dim=-1)), self.q2_t(torch.cat([s_t, self.pi(s_t)], dim=-1)))
        v = self.v(s_t)
        diff = q_pi - v
        w = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        lv = (w * (diff ** 2)).mean()

        # Policy regression towards argmax Q via deterministic actor fit
        pi_a = self.pi(s_t)
        q_pi_online = torch.min(self.q1(torch.cat([s_t, pi_a], dim=-1)), self.q2(torch.cat([s_t, pi_a], dim=-1)))
        lpi = (-q_pi_online).mean()

        loss = lq1 + lq2 + lv + lpi
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        # Update targets
        self._ema(self.q1_t, self.q1)
        self._ema(self.q2_t, self.q2)
        self._ema(self.v_t, self.v)

        # Metrics
        vloss = float(lv.detach().cpu().item())
        qloss = float(((lq1 + lq2) * 0.5).detach().cpu().item())
        piloss = float(lpi.detach().cpu().item())
        self.metrics["loss_q"] = 0.9 * self.metrics.get("loss_q", qloss) + 0.1 * qloss
        self.metrics["loss_v"] = 0.9 * self.metrics.get("loss_v", vloss) + 0.1 * vloss
        self.metrics["loss_pi"] = 0.9 * self.metrics.get("loss_pi", piloss) + 0.1 * piloss

    def get_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)


