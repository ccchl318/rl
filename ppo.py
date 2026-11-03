"""PPO 核心实现：Actor-Critic 网络（带动作掩码）、GAE、截断式 PPO 更新、贪心评估。"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from game2048_env import Game2048Env3x3


def _init_linear(m: nn.Linear, gain: float):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.constant_(m.bias, 0.0)


class ActorCritic(nn.Module):
    """策略与价值网络使用各自独立的 MLP（避免梯度相互干扰）。

    输入为 144 维 one-hot 特征；策略头输出 4 个动作 logits，
    非法动作的 logits 被屏蔽为 -1e8（动作掩码）。
    """

    def __init__(self, obs_dim: int = 144, n_actions: int = 4, hidden: int = 256):
        super().__init__()
        self.pi_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.pi_net:
            if isinstance(m, nn.Linear):
                _init_linear(m, np.sqrt(2) if m.out_features != n_actions else 0.01)
        for m in self.v_net:
            if isinstance(m, nn.Linear):
                _init_linear(m, np.sqrt(2) if m.out_features != 1 else 1.0)

    def policy_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.pi_net(obs)
        return torch.where(mask, logits, torch.full_like(logits, -1e8))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v_net(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        dist = Categorical(logits=self.policy_logits(obs, mask))
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)


def compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    """广义优势估计 (GAE-lambda)。

    rewards/values: np.ndarray [T, N]；dones: [T, N]；last_values: [N]（bootstrap）。
    返回 adv, ret（均为 [T, N]）。
    """
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    return adv, adv + values


def ppo_update(model: ActorCritic, optimizer, batch: dict, clip_eps: float,
               epochs: int, minibatch_size: int, vf_coef: float, ent_coef: float,
               max_grad_norm: float = 0.5):
    """对一份 rollout 数据做 PPO-clip 更新，返回统计量字典。"""
    stats = defaultdict(float)
    n = batch["obs"].shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["obs"].device)
        for idx in torch.split(perm, minibatch_size):
            obs = batch["obs"][idx]
            mask = batch["mask"][idx]
            act = batch["act"][idx]
            old_logp = batch["logp"][idx]
            adv = batch["adv"][idx]
            ret = batch["ret"][idx]
            old_v = batch["val"][idx]

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 优势标准化

            dist = Categorical(logits=model.policy_logits(obs, mask))
            new_logp = dist.log_prob(act)
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            pi_loss = -torch.min(surr1, surr2).mean()

            v = model.value(obs)
            v_clipped = old_v + torch.clamp(v - old_v, -clip_eps, clip_eps)
            v_loss = 0.5 * torch.max((v - ret) ** 2, (v_clipped - ret) ** 2).mean()

            ent = dist.entropy().mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl"] += (old_logp - new_logp).mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > clip_eps).float().mean().item()
                stats["n_updates"] += 1
    k = max(stats["n_updates"], 1)
    return {key: val / k for key, val in stats.items() if key != "n_updates"}


@torch.no_grad()
def run_greedy_episodes(model: ActorCritic, n_episodes: int, seed: int = 12345,
                        device: str = "cpu", record: bool = False):
    """贪心（argmax）策略评估。返回统计量；record=True 时同时记录最优一局轨迹。"""
    model.eval()
    env = Game2048Env3x3()
    results = defaultdict(list)
    best = {"board_sum": -1}
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        traj = [] if record else None
        done = False
        length = 0
        while not done:
            mask = env.action_mask()
            o = torch.from_numpy(obs).unsqueeze(0).to(device)
            m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
            action, _ = model.act(o, m, deterministic=True)
            a = int(action.item())
            if record:
                traj.append((info["board"], a))
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            length += 1
        results["score"].append(info["score"])
        results["max_tile"].append(info["max_tile"])
        results["board_sum"].append(info["board_sum"])
        results["length"].append(length)
        if record and info["board_sum"] > best["board_sum"]:
            best = {"board_sum": info["board_sum"], "score": info["score"],
                    "max_tile": info["max_tile"], "traj": traj}
    out = {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}
    for t in (128, 256, 512, 1024):
        out[f"p>={t}"] = (float(np.mean([m >= t for m in results["max_tile"]])), 0.0)
    if record:
        out["best_episode"] = best
        out["raw"] = {k: list(v) for k, v in results.items()}
    return out
