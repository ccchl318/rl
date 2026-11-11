"""Afterstate 表示与模型期望优势（PPO 的 model-based 变体）。

afterstate = 滑动后、生成新块前的棋盘，是 (状态, 动作) 的确定函数。
W(s') = E[回报 | s']（spawn 期望被边缘化进 W 的定义）。
优势用已知模型精确计算，零 spawn 采样噪声：
    Q(s,a)  = r(s,a) + gamma * W(psi(s,a))
    V_pi(s) = sum_a pi(a|s) * Q(s,a)
    A(s,a)  = Q(s,a) - V_pi(s)
策略更新仍是 PPO clip 目标（actor 不动），critic 用 afterstate TD 单独训练。
"""
import numpy as np
import torch
import torch.nn.functional as F

from game2048_env_fast import (GAIN, LINE_IDX, MAX_LOG2, MOVE_L, NUM_CELLS,
                               NUM_CHANNELS, OBS_DIM)


def cells_from_obs(obs):
    """(N,144) one-hot -> (N,9) 等级（0=空，k=2^k）。"""
    return obs.view(-1, NUM_CELLS, NUM_CHANNELS).argmax(dim=-1)


def obs_from_cells(cells):
    """(N,9) 等级 -> (N,144) one-hot float32。"""
    cells = cells.clamp(0, MAX_LOG2)
    n = cells.shape[0]
    obs = torch.zeros((n, OBS_DIM), dtype=torch.float32)
    idx = torch.arange(NUM_CELLS)[None, :] * NUM_CHANNELS + cells
    obs[torch.arange(n)[:, None].expand_as(idx), idx] = 1.0
    return obs


def apply_actions(cells, actions):
    """对 (N,9) 等级棋盘执行 (N,) 动作，返回 (after, gain, moved)，不生成新块。"""
    n = cells.shape[0]
    rows = np.arange(n)
    idxp = LINE_IDX[actions]                     # (N,3,3)
    c = cells[rows[:, None, None], idxp]         # (N,3,3)
    codes = (c[..., 0] << 8) | (c[..., 1] << 4) | c[..., 2]
    new_codes = MOVE_L[codes]
    gain = GAIN[codes].sum(axis=1)
    moved = (new_codes != codes).any(axis=1)
    unpacked = np.stack([new_codes >> 8 & 15, new_codes >> 4 & 15,
                         new_codes & 15], axis=-1)
    after = np.zeros((n, NUM_CELLS), dtype=np.int64)
    after[rows[:, None, None], idxp] = unpacked
    return after, gain, moved


def train_critic(model, optimizer, aft_obs, targets, epochs, batch, device):
    """afterstate TD：W(s') 回归到 r + gamma*W(s'_{next})。"""
    n = aft_obs.shape[0]
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.mse_loss(model.v_net(aft_obs[idx]).squeeze(-1), targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def model_advantages(model, buffer, args, device, last_cells):
    """用模型精确计算优势 A(s,a) = Q - V_pi（零 spawn 采样）。

    buffer: RolloutBuffer（obs mask rew done act [T,N]）；
    last_cells: (N,9) roll-out 结束时的棋盘（bootstrap 用）。
    返回 (adv (T*N,), ret (T*N,), aft_obs, w_target, critic_loss 占位)。
    """
    T, N = buffer.steps, buffer.n_envs
    M = T * N
    dev = device
    cells = cells_from_obs(buffer.obs.view(M, -1))
    cells_np = cells.cpu().numpy()
    act_np = buffer.act.view(-1).cpu().numpy()
    rew = torch.from_numpy(buffer.rew.reshape(-1)).float().to(dev)
    done = torch.from_numpy(buffer.done.reshape(-1)).float().to(dev)

    after_np, gain_np, moved_np = apply_actions(cells_np, act_np)
    aft_obs = obs_from_cells(torch.from_numpy(after_np)).to(dev)      # (M,144)
    w_after = model.v_net(aft_obs).squeeze(-1)                        # W(s'_t)

    # 4 个动作的 afterstate（V_pi 需要）
    cells4 = np.repeat(cells_np, 4, axis=0)
    acts4 = np.tile(np.arange(4), M)
    after4_np, gain4_np, moved4_np = apply_actions(cells4, acts4)
    w4 = model.v_net(obs_from_cells(torch.from_numpy(after4_np)).to(dev)).squeeze(-1).view(M, 4)
    w_state = model.v_net(obs_from_cells(cells).to(dev)).squeeze(-1)   # 无效动作的 afterstate = 原棋盘
    w_last = model.v_net(obs_from_cells(torch.from_numpy(last_cells)).to(dev)).squeeze(-1)

    # 策略分布 pi(a|s)（当前 actor，mask 生效）
    logits = model.pi_net(buffer.obs.view(M, -1).to(dev))
    mask = buffer.mask.view(M, 4).to(dev)
    logits = torch.where(mask, logits, torch.full_like(logits, -1e8))
    probs = torch.softmax(logits, dim=-1)

    scale = args.reward_scale
    moved4 = torch.from_numpy(moved4_np).to(dev).view(M, 4)
    gain4 = torch.from_numpy(gain4_np).to(dev).view(M, 4)
    q4 = torch.where(moved4,
                     gain4 * scale + args.gamma * w4,
                     torch.full_like(w4, -1.0 * scale) + args.gamma * w_state[:, None])
    v_pi = (probs * q4).sum(-1)
    q_taken = rew + args.gamma * w_after
    adv_model = (q_taken - v_pi)
    # 标准化到与 PPO 采样优势同尺度（确定性优势方向一致，需软缩放）
    adv_model = (adv_model - adv_model.mean()) / (adv_model.std() + 1e-8)
    adv = (adv_model * args.adv_scale).cpu().numpy()

    # 批评家目标：W(s'_t) ← r_t + gamma*W(s'_{t+1})，终局不 bootstrap
    next_w = torch.empty(M, device=dev)
    next_w[:-N] = w_after[N:]
    next_w[-N:] = w_last
    w_target = rew + args.gamma * next_w * (1.0 - done)
    # 目标裁剪：防止自举发散（v_loss 爆炸的直接原因）
    w_clip = args.critic_target_clip
    w_target = w_after.detach() + (w_target - w_after.detach()).clamp(-w_clip, w_clip)
    return adv, q_taken.cpu().numpy(), aft_obs, w_target.detach()
