"""CUDA Graph 加速：rollout 推理图 + PPO 更新图。

背景：144→256→256 的小网络 + batch 2048 的小批次下，GPU 上的耗时几乎全是
kernel 启动/派发开销与 .item() 强制同步，真实计算 <1%。本模块用 CUDA Graph
把每时间步推理与每 minibatch 更新压缩为一次图重放。

设计要点：
- 图内不做任何随机抽样（Categorical 采样在 CPU 上用 numpy 完成，分布与
  torch.distributions.Categorical 等价），避免 RNG 捕获/重放的坑；
- 图输入输出全部是预分配的静态张量：rollout 每步只有一次 H2D（obs+mask,
  36KB，pinned 中转）与一次 D2H（logits+value, 1.25KB），无 .item() 同步；
- 更新图捕获 gather→loss→backward；梯度裁剪与 fused Adam 在图外执行
  （lr 退火因此天然动态）；ent_coef 是图内读取的静态标量张量，每迭代填充，
  不被烘焙进图；
- 统计量（pi_loss/v_loss/entropy/kl/clip_frac）在图内累加到静态缓冲，
  全部 minibatch 重放完后一次性 D2H。

数值语义与 ppo.py 逐式等价（同样的损失/裁剪/GAE/Adam），仅浮点求和顺序
与 fused Adam 的实现差异带来的 ~1e-5 级别数值差。
"""

import numpy as np
import torch

# 推理输出: logits+value 拼在一起做一次 D2H
_LOGIT_DIM = 4


def _capture(warmup_fn, capture_fn):
    """按 PyTorch 官方流程：旁路流预热 3 次 -> 捕获。返回 (graph, output)。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            warmup_fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = capture_fn()
    return g, out


class GraphedPolicy:
    """批量贪心/采样推理的 CUDA Graph 封装。

    infer(obs_np, mask_np) 采样版 -> (act, logp, val) CPU numpy
    value_only(obs_np)     -> val CPU numpy（bootstrap 用）
    采样在 CPU 完成：softmax -> 逆变换抽样（分布同 Categorical，随机流不同）。
    """

    def __init__(self, model, n_envs: int, device, seed: int = 0,
                 obs_dim: int = 144):
        self.model = model
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.static_obs = torch.zeros(n_envs, obs_dim, device=device)
        self.static_mask = torch.zeros(n_envs, _LOGIT_DIM, dtype=torch.bool,
                                       device=device)
        self.static_out = torch.zeros(n_envs, _LOGIT_DIM + 1, device=device)
        self.pinned_in = torch.zeros(n_envs, obs_dim).pin_memory()
        self.pinned_mask = torch.zeros(n_envs, _LOGIT_DIM,
                                       dtype=torch.bool).pin_memory()
        self.pinned_out = torch.zeros(n_envs, _LOGIT_DIM + 1).pin_memory()

        def warmup():
            self._forward()

        def capture():
            return self._forward()

        self.graph, _ = _capture(warmup, capture)

    @torch.no_grad()
    def _forward(self):
        obs, mask = self.static_obs, self.static_mask
        logits_raw = self.model.pi_net(obs)
        logits = torch.where(mask, logits_raw, torch.full_like(logits_raw, -1e8))
        value = self.model.v_net(obs).squeeze(-1)
        self.static_out[:, :_LOGIT_DIM] = logits
        self.static_out[:, _LOGIT_DIM] = value
        return self.static_out

    @torch.no_grad()
    def infer(self, obs_np: np.ndarray, mask_np: np.ndarray):
        """填 pinned -> H2D -> 图重放 -> D2H -> CPU 采样。"""
        self.pinned_in.copy_(torch.from_numpy(obs_np))
        self.pinned_mask.copy_(torch.from_numpy(mask_np))
        self.static_obs.copy_(self.pinned_in, non_blocking=True)
        self.static_mask.copy_(self.pinned_mask, non_blocking=True)
        self.graph.replay()
        self.pinned_out.copy_(self.static_out)
        out = self.pinned_out.numpy()
        logits, val = out[:, :_LOGIT_DIM], out[:, _LOGIT_DIM]

        # CPU 采样（mask 中非法动作 logits=-1e8，softmax 后概率为 0）
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        cum = np.cumsum(p, axis=1)
        u = self.rng.random(self.n)
        act = (cum < u[:, None]).sum(axis=1)
        act = np.clip(act, 0, _LOGIT_DIM - 1)
        logp = np.log(np.maximum(p[np.arange(self.n), act], 1e-30)).astype(np.float32)
        return act.astype(np.int64), logp, val.astype(np.float32)

    @torch.no_grad()
    def value_only(self, obs_np: np.ndarray) -> np.ndarray:
        self.pinned_in.copy_(torch.from_numpy(obs_np))
        self.static_obs.copy_(self.pinned_in, non_blocking=True)
        self.graph.replay()
        self.pinned_out.copy_(self.static_out)
        return self.pinned_out.numpy()[:, _LOGIT_DIM].astype(np.float32)


class GraphedUpdater:
    """PPO-clip 更新的 CUDA Graph 封装（fwd+bwd 在图内，clip+Adam 在图外）。

    要求 n_total % minibatch == 0（形状完全静态）；不满足时请退回 ppo.ppo_update。
    update(batch_cpu, ent_coef) 的数学语义与 ppo.ppo_update 一致：
    优势标准化、ratio-clip、value-clip、熵正则、clip_grad_norm_、Adam。
    """

    def __init__(self, model, optimizer, n_total: int, minibatch: int,
                 vf_coef: float, clip_eps: float, device, epochs: int,
                 max_grad_norm: float = 0.5, obs_dim: int = 144):
        assert n_total % minibatch == 0, "GraphedUpdater 要求 n_total 被 minibatch 整除"
        self.model, self.opt = model, optimizer
        self.n, self.mb, self.epochs = n_total, minibatch, epochs
        self.vf_coef, self.clip_eps, self.max_norm = vf_coef, clip_eps, max_grad_norm
        self.device = device

        def mk(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.b = {k: mk(n_total, obs_dim) if k == "obs" else
                  mk(n_total, 4, dtype=torch.bool) if k == "mask" else
                  mk(n_total, dtype=torch.long) if k == "act" else
                  mk(n_total)
                  for k in ("obs", "mask", "act", "logp", "val", "adv", "ret")}
        self.idx = mk(minibatch, dtype=torch.long)
        self.ent_t = torch.zeros((), device=device)  # 0 维标量：每迭代填充，不被烘焙
        self.acc = mk(5)  # pi_loss, v_loss, entropy, kl, clip_frac 的累计

        # 预分配 .grad（静态缓冲），供图内 backward 原地写入
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        def warmup():
            self._step()

        def capture():
            return self._step()

        self.graph, _ = _capture(warmup, capture)

    def _step(self):
        """一次 minibatch 更新（图体）：gather -> 损失 -> backward -> 统计累加。"""
        idx = self.idx
        obs = self.b["obs"][idx]
        mask = self.b["mask"][idx]
        act = self.b["act"][idx]
        old_logp = self.b["logp"][idx]
        adv = self.b["adv"][idx]
        ret = self.b["ret"][idx]
        old_v = self.b["val"][idx]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logits_raw = self.model.pi_net(obs)
        logits = torch.where(mask, logits_raw, torch.full_like(logits_raw, -1e8))
        z = logits - logits.max(dim=-1, keepdim=True).values
        logp_all = z - torch.logsumexp(z, dim=-1, keepdim=True)
        new_logp = logp_all.gather(1, act.unsqueeze(1)).squeeze(1)
        p = torch.exp(logp_all)
        ent = -(p * logp_all).sum(-1).mean()

        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        pi_loss = -torch.min(surr1, surr2).mean()

        v = self.model.v_net(obs).squeeze(-1)
        v_clipped = old_v + torch.clamp(v - old_v, -self.clip_eps, self.clip_eps)
        v_loss = 0.5 * torch.max((v - ret) ** 2, (v_clipped - ret) ** 2).mean()

        loss = pi_loss + self.vf_coef * v_loss - self.ent_t * ent
        self.opt.zero_grad(set_to_none=False)
        loss.backward()

        kl = (old_logp - new_logp).mean()
        clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
        # detach：统计缓冲不能带上梯度（否则 acc 被标 requires_grad，无法 .cpu()）
        self.acc.add_(torch.stack([pi_loss.detach(), v_loss.detach(),
                                   ent.detach(), kl.detach(), clip_frac.detach()]))
        return loss

    def update(self, batch_cpu: dict, ent_coef: float):
        """batch_cpu: 与 ppo.ppo_update 相同的 dict（CPU 张量）。返回均值统计。"""
        for k, t in self.b.items():
            t.copy_(batch_cpu[k])  # 每迭代一次集中 H2D
        self.ent_t.fill_(ent_coef)
        self.acc.zero_()

        n_mb = self.n // self.mb
        for _ in range(self.epochs):
            perm = torch.randperm(self.n, device=self.device)
            for k in range(n_mb):
                self.idx.copy_(perm[k * self.mb:(k + 1) * self.mb])
                self.graph.replay()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.opt.step()
        stats = self.acc.cpu().numpy() / (n_mb * self.epochs)
        return {"pi_loss": float(stats[0]), "v_loss": float(stats[1]),
                "entropy": float(stats[2]), "kl": float(stats[3]),
                "clip_frac": float(stats[4])}
