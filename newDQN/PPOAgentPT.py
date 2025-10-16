import os
from collections import deque
from typing import Dict

import numpy as np

try:
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    import gym  # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    return np.array([
        sum(discount ** i * x[idx + i] for i in range(len(x) - idx))
        for idx in range(len(x))
    ], dtype=np.float32)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.pi = nn.Linear(64, act_dim)
        self.v = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        logits = self.pi(x)
        value = self.v(x)
        return logits, value


class PPOBuffer:
    def __init__(self, obs_dim: int, size: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int64)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, np.ndarray]:
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        self.ptr, self.path_start_idx = 0, 0
        return data


class PPOAgent:
    def __init__(self, env, clip_ratio: float = 0.2, pi_lr: float = 3e-4, vf_lr: float = 1e-3,
                 train_iters: int = 80, target_kl: float = 0.01, device: str | None = None):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ac = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.pi_opt = optim.Adam(self.ac.parameters(), lr=pi_lr)
        self.vf_opt = optim.Adam(self.ac.parameters(), lr=vf_lr)

    def _compute_loss(self, obs_t, act_t, adv_t, ret_t, logp_old_t):
        logits, value = self.ac(obs_t)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_t)
        ratio = torch.exp(logp - logp_old_t)
        min_adv = torch.where(adv_t > 0.0, (1.0 + self.clip_ratio) * adv_t, (1.0 - self.clip_ratio) * adv_t)
        pi_loss = -torch.mean(torch.minimum(ratio * adv_t, min_adv))
        v_loss = torch.mean((value.squeeze(-1) - ret_t) ** 2)
        entropy = torch.mean(dist.entropy())
        total_loss = pi_loss + 0.5 * v_loss - 0.01 * entropy
        approx_kl = torch.mean(logp_old_t - logp)
        return total_loss, pi_loss, v_loss, entropy, approx_kl

    def update(self, data):
        obs_t = to_tensor(data['obs'], self.device)
        act_t = torch.as_tensor(data['act'], dtype=torch.long, device=self.device)
        adv_t = to_tensor(data['adv'], self.device)
        ret_t = to_tensor(data['ret'], self.device)
        logp_old_t = to_tensor(data['logp'], self.device)
        for _ in range(self.train_iters):
            total_loss, pi_loss, v_loss, entropy, kl = self._compute_loss(obs_t, act_t, adv_t, ret_t, logp_old_t)
            self.pi_opt.zero_grad(set_to_none=True)
            total_loss.backward()
            self.pi_opt.step()
            if kl.item() > 1.5 * self.target_kl:
                break

    def select_action(self, obs: np.ndarray):
        obs_t = to_tensor(obs[np.newaxis, ...], self.device)
        with torch.no_grad():
            logits, value = self.ac(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(value[0, 0].item()), float(logp[0].item())

    def train(self, epochs: int = 50, steps_per_epoch: int = 4000, model_dir: str = "./models"):
        os.makedirs(model_dir, exist_ok=True)
        buffer = PPOBuffer(self.obs_dim, steps_per_epoch)
        returns = deque(maxlen=100)
        for _ in range(epochs):
            reset_out = self.env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            ep_ret = 0.0
            for t in range(steps_per_epoch):
                act, val, logp = self.select_action(obs)
                step_out = self.env.step(act)
                if len(step_out) == 5:
                    next_obs, rew, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                else:
                    next_obs, rew, done, _ = step_out
                buffer.store(obs, act, rew, val, logp)
                obs = next_obs
                ep_ret += rew
                if done or (t == steps_per_epoch - 1):
                    with torch.no_grad():
                        last_val = 0.0 if done else float(self.ac(to_tensor(obs[np.newaxis, ...], self.device))[1][0, 0].item())
                    buffer.finish_path(last_val)
                    returns.append(ep_ret)
                    reset_out = self.env.reset()
                    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    ep_ret = 0.0
            data = buffer.get()
            self.update(data)
        # Optionnel: sauvegarde
        try:
            torch.save(self.ac.state_dict(), os.path.join(model_dir, 'ppo_pt_actor_critic.pt'))
        except Exception:
            pass


def train_ppo_with_env(env, epochs: int = 50, steps_per_epoch: int = 4000):
    agent = PPOAgent(env)
    agent.train(epochs=epochs, steps_per_epoch=steps_per_epoch)
    return agent


