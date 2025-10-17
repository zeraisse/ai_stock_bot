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


class Callback:
    def on_train_begin(self, agent, info: dict):
        pass
    def on_epoch_end(self, agent, epoch: int, logs: dict):
        pass
    def on_train_end(self, agent, history: list):
        pass


class EarlyStopping(Callback):
    def __init__(self, monitor: str = 'mean_return', patience: int = 20, mode: str = 'max'):
        self.monitor = monitor
        self.patience = int(patience)
        self.mode = mode
        self.best = -np.inf if mode == 'max' else np.inf
        self.wait = 0
        self.stopped_epoch = None
        self.should_stop = False
    def on_epoch_end(self, agent, epoch: int, logs: dict):
        val = logs.get(self.monitor)
        if val is None:
            return
        improved = (val > self.best) if self.mode == 'max' else (val < self.best)
        if improved:
            self.best = val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch


class BestModelSaver(Callback):
    def __init__(self, path: str = './models/ppo_best.pt', monitor: str = 'mean_return', mode: str = 'max'):
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.best = -np.inf if mode == 'max' else np.inf
    def on_epoch_end(self, agent, epoch: int, logs: dict):
        val = logs.get(self.monitor)
        if val is None:
            return
        improved = (val > self.best) if self.mode == 'max' else (val < self.best)
        if improved:
            self.best = val
            try:
                os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
                torch.save(agent.ac.state_dict(), self.path)
            except Exception:
                pass


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
    def __init__(self, env, clip_ratio: float = 0.2, pi_lr: float = 1e-4, vf_lr: float = 1e-3,
                 train_iters: int = 60, target_kl: float = 0.01, device: str | None = None,
                 minibatch_size: int = 256):
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
        # Interpret train_iters as number of update epochs when using minibatches
        self.minibatch_size = int(minibatch_size)

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
        obs = data['obs']
        acts = data['act']
        advs = data['adv']
        rets = data['ret']
        logps = data['logp']
        N = len(acts)
        pi_losses = []
        v_losses = []
        entropies = []
        kls = []
        iters_done = 0
        for _ in range(self.train_iters):  # treat as "update epochs"
            idx = np.random.permutation(N)
            for start in range(0, N, self.minibatch_size):
                end = min(start + self.minibatch_size, N)
                mb = idx[start:end]
                obs_t = to_tensor(obs[mb], self.device)
                act_t = torch.as_tensor(acts[mb], dtype=torch.long, device=self.device)
                adv_t = to_tensor(advs[mb], self.device)
                ret_t = to_tensor(rets[mb], self.device)
                logp_old_t = to_tensor(logps[mb], self.device)
                total_loss, pi_loss, v_loss, entropy, kl = self._compute_loss(obs_t, act_t, adv_t, ret_t, logp_old_t)
                self.pi_opt.zero_grad(set_to_none=True)
                total_loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                except Exception:
                    pass
                self.pi_opt.step()
                pi_losses.append(float(pi_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(entropy.item()))
                kls.append(float(kl.item()))
                iters_done += 1
                if kl.item() > 1.5 * self.target_kl:
                    break
            if len(kls) and kls[-1] > 1.5 * self.target_kl:
                break
        # return summary stats for logging
        stats = {
            'pi_loss': float(np.mean(pi_losses)) if len(pi_losses) else 0.0,
            'v_loss': float(np.mean(v_losses)) if len(v_losses) else 0.0,
            'entropy': float(np.mean(entropies)) if len(entropies) else 0.0,
            'kl': float(np.mean(kls)) if len(kls) else 0.0,
            'iters': iters_done,
        }
        return stats

    def select_action(self, obs: np.ndarray):
        obs_t = to_tensor(obs[np.newaxis, ...], self.device)
        with torch.no_grad():
            logits, value = self.ac(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(value[0, 0].item()), float(logp[0].item())

    def train(self, epochs: int = 50, steps_per_epoch: int = 4000, model_dir: str = "./models", callbacks: list | None = None):
        os.makedirs(model_dir, exist_ok=True)
        buffer = PPOBuffer(self.obs_dim, steps_per_epoch)
        returns = deque(maxlen=100)
        training_history = []  # mean episode return per epoch
        cbs = callbacks or []
        try:
            for cb in cbs:
                cb.on_train_begin(self, {'epochs': epochs, 'steps_per_epoch': steps_per_epoch})
        except Exception:
            pass
        for epoch in range(epochs):
            reset_out = self.env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            ep_ret = 0.0
            ep_returns_epoch = []
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
                    ep_returns_epoch.append(float(ep_ret))
                    reset_out = self.env.reset()
                    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    ep_ret = 0.0
            data = buffer.get()
            upd_stats = self.update(data)
            # record epoch mean return (0 if no full episode finished)
            epoch_mean = float(np.mean(ep_returns_epoch)) if len(ep_returns_epoch) > 0 else 0.0
            training_history.append(epoch_mean)
            logs = {
                'mean_return': epoch_mean,
                **upd_stats,
            }
            try:
                for cb in cbs:
                    cb.on_epoch_end(self, epoch, logs)
            except Exception:
                pass
            try:
                print(
                    f"[PPO] Epoch {epoch + 1}/{epochs} | mean_return={epoch_mean:.3f} | "
                    f"pi_loss={upd_stats['pi_loss']:.4f} v_loss={upd_stats['v_loss']:.4f} "
                    f"entropy={upd_stats['entropy']:.4f} kl={upd_stats['kl']:.5f} "
                    f"iters={upd_stats['iters']} | steps={steps_per_epoch}"
                )
            except Exception:
                pass
            try:
                for cb in cbs:
                    if isinstance(cb, EarlyStopping) and cb.should_stop:
                        raise StopIteration
            except StopIteration:
                break
        # Optionnel: sauvegarde
        try:
            torch.save(self.ac.state_dict(), os.path.join(model_dir, 'ppo_pt_actor_critic.pt'))
        except Exception:
            pass
        try:
            for cb in cbs:
                cb.on_train_end(self, training_history)
        except Exception:
            pass
        return training_history


def train_ppo_with_env(env, epochs: int = 50, steps_per_epoch: int = 4000):
    agent = PPOAgent(env)
    history = agent.train(epochs=epochs, steps_per_epoch=steps_per_epoch)
    return agent, history


