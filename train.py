"""PPO 训练脚本：同步向量化环境采集 -> GAE -> PPO-clip 更新。

用法示例：
    python train.py --run-name main --iters 3000                # 默认 fast 环境 + CPU rollout
    python train.py --env py --device cuda --run-name legacy    # 完全复现旧的逐环境实现
    python train.py --run-name ablation_low_ent --iters 400 --ent-coef 0.002 --ent-final 0.0002

环境选择（--env）：
    fast  向量化查表环境（game2048_env_fast.py，语义与 py 逐位等价，吞吐高一个量级以上）
    py    旧逐环境实现（game2048_env.py），行为与历史实验完全一致
设备（--device 管 PPO 更新；--rollout-device 管 rollout 推理，默认 CPU——144→256→256
的小网络在 CPU 上推理更快，且多进程并发训练时不再挤占 GPU）。

输出（位于 results/<run-name>/）：
    train_log.csv   每次迭代的训练统计（最近 100 局滑窗均值）
    eval_log.csv    周期性贪心评估统计
    last.pt         最新模型；best.pt 按评估指标保存的最优模型
"""

import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch

from game2048_env import Game2048Env3x3, ACTION_MEANING
from game2048_env_fast import FastGame2048Batch
from branch_adv import collect_branch_batch
from afterstate import model_advantages, train_critic
from gpu_graph import GraphedPolicy, GraphedUpdater
from ppo import (ActorCritic, compute_gae, ppo_update, run_greedy_episodes,
                 run_greedy_episodes_fast)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, default="main")
    p.add_argument("--iters", type=int, default=3000, help="迭代次数（每次迭代采集 n_envs*steps 步）")
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=128, help="每个环境每次迭代采集的步数")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--lr-final-ratio", type=float, default=0.1, help="lr 线性退火到初值的该比例")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--minibatch-size", type=int, default=2048)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.02, help="熵系数初始值（线性退火）")
    p.add_argument("--ent-final", type=float, default=0.002, help="熵系数终值")
    p.add_argument("--reward-scale", type=float, default=0.25, help="奖励 = 合并得分 * reward_scale")
    p.add_argument("--shaping", choices=["none", "bonus512", "potential"], default="none",
                   help="奖励塑形：bonus512=首次合成512给额外奖励；potential=max_tile等级势函数塑形")
    p.add_argument("--bonus512", type=float, default=200.0, help="bonus512 模式的额外奖励幅度")
    p.add_argument("--phi-w", type=float, default=1.0, help="potential 模式的势函数权重")
    p.add_argument("--branch-k", type=int, default=0,
                   help="状态级 GRPO：每迭代采样的拥挤状态数 K（0=关闭）")
    p.add_argument("--branch-g", type=int, default=8, help="每状态的分支数 G")
    p.add_argument("--branch-h", type=int, default=32, help="分支续局步数 H")
    p.add_argument("--branch-start-iter", type=int, default=0,
                   help="分支采样从该迭代才激活（前期纯基线 PPO；0=立即激活）")
    p.add_argument("--afterstate-adv", action="store_true",
                   help="afterstate critic + 模型精确优势（PPO 的 model-based 变体）")
    p.add_argument("--critic-epochs", type=int, default=3,
                   help="afterstate critic 每迭代的回归轮数")
    p.add_argument("--critic-lr", type=float, default=6.25e-5,
                   help="afterstate critic 独立学习率（默认 0.1x 主 lr）")
    p.add_argument("--critic-target-clip", type=float, default=5.0,
                   help="critic TD 目标相对当前预估的裁剪幅度")
    p.add_argument("--adv-scale", type=float, default=1.0,
                   help="模型优势缩放")
    p.add_argument("--adv-mix", type=float, default=0.5,
                   help="模型优势的混合权重（1=纯模型优势，0=纯 GAE）")
    p.add_argument("--no-mask", action="store_true", help="消融：不使用动作掩码")
    p.add_argument("--env", choices=["fast", "py"], default="fast",
                   help="fast=向量化查表环境（默认）；py=旧逐环境实现（完全复现旧行为）")
    p.add_argument("--rollout-device", type=str, default="cpu",
                   help="rollout 推理设备（仅 --no-graph 或 CPU 训练时生效；默认 CUDA 图路径下推理在 GPU 重放）")
    p.add_argument("--no-graph", action="store_true",
                   help="禁用 CUDA Graph（推理图+更新图），退回逐算子执行")
    p.add_argument("--extend-iters", type=int, default=0,
                   help="主调度结束后的精修迭代数（lr 恒定 --extend-lr，熵固定在终值）")
    p.add_argument("--extend-lr", type=float, default=6e-5,
                   help="精修阶段的恒定学习率")
    p.add_argument("--extend2-iters", type=int, default=0,
                   help="第三阶段（超精修）迭代数，lr 更低")
    p.add_argument("--extend2-lr", type=float, default=2e-5,
                   help="第三阶段的恒定学习率")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="权重 EMA 衰减系数（0=关闭）")
    p.add_argument("--ema-decays", type=str, default="",
                   help="多 EMA 衰减系数，逗号分隔（如 0.999,0.9995）；同时维护多个并取最优")
    p.add_argument("--extend3-iters", type=int, default=0,
                   help="第四阶段（更低 lr）迭代数")
    p.add_argument("--extend3-lr", type=float, default=5e-6,
                   help="第四阶段恒定学习率")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    p.add_argument("--init-from", type=str, default=None,
                   help="从 checkpoint 载入模型权重继续训练（优化器与 lr/熵调度全新开始；用于微调/复热）")
    return p.parse_args()


class RolloutBuffer:
    """按 [T, N] 布局存储一次迭代的轨迹数据。"""

    def __init__(self, n_envs: int, steps: int, obs_dim: int, device):
        self.obs = torch.zeros(steps, n_envs, obs_dim, device=device)
        self.mask = torch.zeros(steps, n_envs, 4, dtype=torch.bool, device=device)
        self.act = torch.zeros(steps, n_envs, dtype=torch.long, device=device)
        self.logp = torch.zeros(steps, n_envs, device=device)
        self.val = torch.zeros(steps, n_envs, device=device)
        self.rew = np.zeros((steps, n_envs), dtype=np.float32)
        self.done = np.zeros((steps, n_envs), dtype=np.float32)
        self.steps, self.n_envs = steps, n_envs

    def flat(self, adv: np.ndarray, ret: np.ndarray) -> dict:
        t, n = self.steps, self.n_envs
        return {
            "obs": self.obs.reshape(t * n, -1),
            "mask": self.mask.reshape(t * n, 4),
            "act": self.act.reshape(t * n),
            "logp": self.logp.reshape(t * n),
            "val": self.val.reshape(t * n),
            "adv": torch.from_numpy(adv.reshape(-1)).to(self.obs.device),
            "ret": torch.from_numpy(ret.reshape(-1)).to(self.obs.device),
        }


def _phi_dense(cells):
    """稠密结构势函数（每步都变化）：空格数、最大等级、单调性、平滑度。

    特征与 expectimax 启发式同族（系数即其搜索评估权重），量级 ~0-30。
    """
    empty = (cells == 0).sum(axis=1).astype(np.float64)
    logs = cells.astype(np.float64)
    grid = logs.reshape(-1, 3, 3)
    dh = (grid[:, :, 1:] - grid[:, :, :-1]).reshape(len(logs), -1)
    dv = (grid[:, 1:, :] - grid[:, :-1, :]).reshape(len(logs), -1)
    inc = np.maximum(dh, 0).sum(1) + np.maximum(dv, 0).sum(1)
    dec = np.maximum(-dh, 0).sum(1) + np.maximum(-dv, 0).sum(1)
    mono = -np.minimum(inc, dec)
    smooth = -(np.abs(dh).sum(1) + np.abs(dv).sum(1))
    return 2.7 * np.log(empty + 1.0) + logs.max(axis=1) + 1.0 * mono + 0.1 * smooth


def apply_shaping(rew, pre_cells, post_cells, done, args):
    """在本步奖励上叠加塑形项（仅对未结束的对局生效）。

    potential: F = gamma * phi(s') - phi(s)（Ng et al. 1999 势函数形式，
    理论上不改变最优策略）；phi 为稠密结构势，每步都变化，把远期大奖励
    的信用提前分摊到每一步，绕开 GAE 的 (gamma*lam)^k 信用衰减。
    bonus512: 首次合成 512 给额外奖励（稀疏对照）。
    """
    alive = done == 0
    if args.shaping == "potential":
        return rew + args.phi_w * (args.gamma * _phi_dense(post_cells) - _phi_dense(pre_cells)) * alive
    return rew + ((post_cells.max(axis=1) >= 9) & (pre_cells.max(axis=1) < 9)) * alive * args.bonus512


def collect_branch_states(model, buffer, args, device):
    """状态级 GRPO：优先选含 256 的拥挤状态（不足则按 等级*100+占用 降级挑选），
    分支采样生成组内相对优势的首动作转移；无候选时返回 None。"""
    obs_all = buffer.obs
    T, N = obs_all.shape[:2]
    cells = obs_all.view(T * N, 9, 16).argmax(dim=-1)
    maxlv = cells.max(dim=1).values
    occ = (cells > 0).sum(dim=1)
    strict = ((maxlv >= 8) & (occ >= 7)).nonzero(as_tuple=True)[0]
    if len(strict) >= args.branch_k:
        pick = strict[torch.randperm(len(strict), device=strict.device)[:args.branch_k]]
    else:
        key = maxlv * 100 + occ
        order = torch.argsort(key, descending=True)
        if len(order) >= args.branch_k:
            pick = order[:args.branch_k]
        else:
            pick = order[torch.randint(0, len(order), (args.branch_k,), device=order.device)]
    cells_list = cells[pick].cpu().numpy()
    model_dev = str(next(model.parameters()).device)
    return collect_branch_batch(model, cells_list, args.branch_g, args.branch_h,
                                model_dev, args.gamma)


def main():
    args = parse_args()
    out_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)
    rollout_device = torch.device(args.rollout_device)
    if args.env == "fast":
        benv = FastGame2048Batch(args.n_envs, seed=args.seed,
                                 reward_scale=args.reward_scale)
        envs = None
    else:
        envs = [Game2048Env3x3() for _ in range(args.n_envs)]
        obs_list, mask_list = [], []
        for i, env in enumerate(envs):
            obs, _ = env.reset(seed=args.seed * 10_000 + i)
            obs_list.append(obs)
            mask_list.append(env.action_mask())

    model = ActorCritic(hidden=args.hidden).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device)
        model.load_state_dict(sd["model"])
        print(f"[{args.run_name}] 从 {args.init_from} 载入模型权重 (来源 it={sd.get('it')})，"
              f"优化器与调度全新开始", flush=True)
    # CUDA 图路径：fast 环境 + GPU 更新 + minibatch 整除（形状全静态才能捕获）
    use_graph = (args.env == "fast" and device.type == "cuda" and not args.no_graph
                 and (args.n_envs * args.steps) % args.minibatch_size == 0)
    ema_decs = ([float(x) for x in args.ema_decays.split(',') if x.strip()]
                if args.ema_decays.strip() else
                ([args.ema_decay] if args.ema_decay > 0 else []))
    emas = {d: {k: v.detach().clone().float() for k, v in model.state_dict().items()}
            for d in ema_decs}
    critic_opt = (torch.optim.Adam(model.v_net.parameters(), lr=args.critic_lr)
                  if args.afterstate_adv else None)
    if use_graph:
        # fused Adam：图外每次 minibatch 一次融合 kernel；lr 退火仍然每迭代生效
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)
        buffer_device = torch.device("cpu")  # rollout 期间 buffer 常驻 CPU，迭代末一次 H2D
        roller = GraphedPolicy(model, args.n_envs, device, seed=args.seed + 1)
        vf_coef_now = 0.0 if args.afterstate_adv else args.vf_coef
        updater = GraphedUpdater(model, optimizer, args.n_envs * args.steps,
                                 args.minibatch_size, vf_coef_now, args.clip,
                                 device, epochs=args.epochs)
        branch_updater = None
        if args.branch_k > 0:
            assert (args.branch_k * args.branch_g) % args.minibatch_size == 0,                 "分支行数必须被 minibatch 整除"
            branch_updater = GraphedUpdater(model, optimizer, args.branch_k * args.branch_g,
                                            args.minibatch_size, vf_coef_now, args.clip,
                                            device, epochs=args.epochs)
        mask_all = np.ones((args.n_envs, 4), dtype=bool)  # --no-mask 用的全真掩码
        rollout_model = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        buffer_device = device
        # fast 路径 rollout 推理设备与更新设备不同时，维护一份推理权重镜像（每次更新后同步）
        if args.env == "fast" and rollout_device.type != device.type:
            rollout_model = ActorCritic(hidden=args.hidden).to(rollout_device)
            rollout_model.eval()
        else:
            rollout_model = None
    obs_dim = envs[0].observation_space.shape[0] if envs is not None else benv.obs.shape[1]
    buffer = RolloutBuffer(args.n_envs, args.steps, obs_dim, buffer_device)

    # 每个环境的局内统计
    ep_score = np.zeros(args.n_envs)
    ep_len = np.zeros(args.n_envs, dtype=int)
    recent = deque(maxlen=100)
    total_steps, total_episodes = 0, 0
    best_metric = -1.0

    train_csv = open(os.path.join(out_dir, "train_log.csv"), "w", newline="")
    train_writer = csv.writer(train_csv)
    train_writer.writerow(["iter", "steps", "episodes", "score_mean", "score_std",
                           "max_tile_mean", "board_sum_mean", "length_mean",
                           "pi_loss", "v_loss", "entropy", "kl", "clip_frac", "lr", "ent_coef"])
    eval_csv = open(os.path.join(out_dir, "eval_log.csv"), "w", newline="")
    eval_writer = csv.writer(eval_csv)
    eval_writer.writerow(["iter", "steps", "score_mean", "score_std", "max_tile_mean",
                          "board_sum_mean", "board_sum_std", "length_mean",
                          "p>=128", "p>=256", "p>=512", "p>=1024",
                          "raw_board_sum", "raw_p512"])

    def save_ckpt(name, it):
        torch.save({"it": it, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "args": vars(args)},
                   os.path.join(out_dir, name))

    total_iters_print = args.iters + args.extend_iters + args.extend2_iters + args.extend3_iters
    print(f"[{args.run_name}] device={device} env={args.env} cuda_graph={'on' if use_graph else 'off'}, "
          f"每迭代 {args.n_envs * args.steps} 步, "
          f"共 {total_iters_print} 迭代 "
          f"(≈{total_iters_print * args.n_envs * args.steps / 1e6:.1f}M 步)"
          + (f"，精修 {args.extend_iters}@{args.extend_lr:g}" if args.extend_iters > 0 else "")
          + (f"，超精修 {args.extend2_iters}@{args.extend2_lr:g}" if args.extend2_iters > 0 else "")
          + (f"，EMA={args.ema_decays or args.ema_decay}" if (args.ema_decays or args.ema_decay > 0) else ""))
    t_start = time.time()
    total_iters = args.iters + args.extend_iters + args.extend2_iters + args.extend3_iters
    for it in range(1, total_iters + 1):
        # 阶段一：线性退火；阶段二/三：恒定低 lr 精修（熵固定终值）
        if it <= args.iters:
            frac = (it - 1) / max(args.iters - 1, 1)
            lr_now = args.lr * (1.0 - (1.0 - args.lr_final_ratio) * frac)
            ent_now = args.ent_coef + (args.ent_final - args.ent_coef) * frac
        elif it <= args.iters + args.extend_iters:
            lr_now = args.extend_lr
            ent_now = args.ent_final
        elif it <= args.iters + args.extend_iters + args.extend2_iters:
            lr_now = args.extend2_lr
            ent_now = args.ent_final
        else:
            lr_now = args.extend3_lr
            ent_now = args.ent_final
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        model.eval()
        with torch.no_grad():
            if use_graph:
                # 图重放路径：buffer 常驻 CPU，无逐时间步 H2D；动作 D2H 仅 1.25KB
                mask_in = benv.mask if not args.no_mask else mask_all
                for t in range(args.steps):
                    act, logp, vals = roller.infer(benv.obs, mask_in)
                    buffer.obs[t] = torch.from_numpy(benv.obs)
                    buffer.mask[t] = torch.from_numpy(mask_in)
                    buffer.act[t] = torch.from_numpy(act)
                    buffer.logp[t] = torch.from_numpy(logp)
                    buffer.val[t] = torch.from_numpy(vals)
                    pre = benv.cells.copy() if args.shaping != "none" else None
                    rew, done = benv.step(act)
                    if args.shaping != "none":
                        rew = apply_shaping(rew, pre, benv.cells, done, args)
                    buffer.rew[t] = rew
                    buffer.done[t] = done
                    for _, stat in benv.pop_finished():
                        recent.append(stat)
                        total_episodes += 1
            elif args.env == "fast":
                infer_model = rollout_model if rollout_model is not None else model
                for t in range(args.steps):
                    obs_batch = torch.from_numpy(benv.obs).to(rollout_device)
                    mask_batch = torch.ones(args.n_envs, 4, dtype=torch.bool,
                                            device=rollout_device) \
                        if args.no_mask else torch.from_numpy(benv.mask).to(rollout_device)
                    act, logp, vals = infer_model.act_and_value(obs_batch, mask_batch)
                    buffer.obs[t] = obs_batch.to(device)
                    buffer.mask[t] = mask_batch.to(device)
                    buffer.act[t] = act.to(device)
                    buffer.logp[t] = logp.to(device)
                    buffer.val[t] = vals.to(device)
                    pre = benv.cells.copy() if args.shaping != "none" else None
                    rew, done = benv.step(act.cpu().numpy())
                    if args.shaping != "none":
                        rew = apply_shaping(rew, pre, benv.cells, done, args)
                    buffer.rew[t] = rew
                    buffer.done[t] = done
                    for _, stat in benv.pop_finished():
                        recent.append(stat)
                        total_episodes += 1
            else:
                for t in range(args.steps):
                    obs_batch = torch.from_numpy(np.stack(obs_list)).to(device)
                    mask_batch = torch.ones(args.n_envs, 4, dtype=torch.bool, device=device) \
                        if args.no_mask else torch.tensor(np.stack(mask_list), dtype=torch.bool, device=device)
                    act, logp = model.act(obs_batch, mask_batch)
                    vals = model.value(obs_batch)

                    buffer.obs[t] = obs_batch
                    buffer.mask[t] = mask_batch
                    buffer.act[t] = act
                    buffer.logp[t] = logp
                    buffer.val[t] = vals

                    act_np = act.cpu().numpy()
                    for i, env in enumerate(envs):
                        obs, rew, term, trunc, info = env.step(int(act_np[i]))
                        buffer.rew[t, i] = rew * args.reward_scale
                        buffer.done[t, i] = float(term or trunc)
                        ep_score[i] += rew
                        ep_len[i] += 1
                        if term or trunc:
                            recent.append((ep_score[i], info["max_tile"],
                                           info["board_sum"], ep_len[i]))
                            total_episodes += 1
                            ep_score[i], ep_len[i] = 0.0, 0
                            obs, _ = env.reset()
                        obs_list[i] = obs
                        mask_list[i] = env.action_mask()
                    total_steps += args.n_envs

        if args.afterstate_adv:
            adv_m, ret, aft_obs, w_target = model_advantages(
                model, buffer, args, device, benv.cells)
            # 与 GAE 按 adv-mix 折中：模型优势零 spawn 噪声但缺采样抖动，
            # 纯用它会使策略过快确定性化（entropy collapse）。折中兼顾两者。
            with torch.no_grad():
                if use_graph:
                    last_values = roller.value_only(benv.obs)
                else:
                    last_values = model.value(
                        torch.from_numpy(benv.obs).to(device)).cpu().numpy()
            adv_g, _ = compute_gae(buffer.rew, buffer.val.cpu().numpy(), buffer.done,
                                   last_values, args.gamma, args.lam)
            adv_g = adv_g.reshape(-1)
            sg = (adv_g - adv_g.mean()) / (adv_g.std() + 1e-8)
            sm = (adv_m - adv_m.mean()) / (adv_m.std() + 1e-8)
            adv = args.adv_mix * sm + (1.0 - args.adv_mix) * sg
            critic_loss = train_critic(model, critic_opt, aft_obs, w_target,
                                       args.critic_epochs, args.minibatch_size, device)
        else:
            # bootstrap 末端价值
            with torch.no_grad():
                if use_graph:
                    last_values = roller.value_only(benv.obs)
                else:
                    obs_final = torch.from_numpy(benv.obs) if args.env == "fast" \
                        else torch.from_numpy(np.stack(obs_list))
                    last_values = model.value(obs_final.to(device)).cpu().numpy()
            adv, ret = compute_gae(buffer.rew, buffer.val.cpu().numpy(), buffer.done,
                                   last_values, args.gamma, args.lam)
    
        model.train()
        if use_graph:
            # 图重放更新：gather->loss->backward 在图内，clip+fused Adam 在图外；
            # batch 仍在 CPU，update() 内部做每迭代一次的集中 H2D
            stats = updater.update(buffer.flat(adv, ret), ent_now)
        else:
            stats = ppo_update(model, optimizer, buffer.flat(adv, ret), args.clip,
                               args.epochs, args.minibatch_size,
                               0.0 if args.afterstate_adv else args.vf_coef, ent_now)
        if args.branch_k > 0 and it >= args.branch_start_iter:
            bflat = collect_branch_states(model, buffer, args, str(device))
            if bflat is not None:
                if use_graph:
                    bstats = branch_updater.update(bflat, ent_now)
                else:
                    bstats = ppo_update(model, optimizer, bflat, args.clip,
                                        max(2, args.epochs // 2), args.minibatch_size,
                                        0.0 if args.afterstate_adv else args.vf_coef, ent_now)
        if rollout_model is not None:
            rollout_model.load_state_dict(model.state_dict())
            rollout_model.eval()
        total_steps = it * args.n_envs * args.steps

        if emas:
            with torch.no_grad():
                cur = model.state_dict()
                for d, st in emas.items():
                    for k, v in cur.items():
                        if v.dtype.is_floating_point:
                            st[k].mul_(d).add_(v.float(), alpha=1 - d)
                        else:
                            st[k] = v.detach().clone()

        if recent:
            arr = np.array(recent)
            tr = [arr[:, 0].mean(), arr[:, 0].std(), arr[:, 1].mean(),
                  arr[:, 2].mean(), arr[:, 3].mean()]
        else:
            tr = [0.0] * 5
        train_writer.writerow([it, total_steps, total_episodes] +
                              [f"{x:.2f}" for x in tr] +
                              [f"{stats['pi_loss']:.4f}", f"{stats['v_loss']:.3f}",
                               f"{stats['entropy']:.3f}", f"{stats['kl']:.4f}",
                               f"{stats['clip_frac']:.3f}", f"{lr_now:.2e}", f"{ent_now:.4f}"])

        if it % 10 == 0 or it == 1:
            ips = total_steps / (time.time() - t_start)
            print(f"it {it:5d} | steps {total_steps / 1e6:6.2f}M | ips {ips:7.0f} | "
                  f"score {tr[0]:7.1f}±{tr[1]:6.1f} | max_tile {tr[2]:6.1f} | "
                  f"sum {tr[3]:6.1f} | len {tr[4]:5.1f} | ent {stats['entropy']:.2f} | "
                  f"kl {stats['kl']:.3f} | v_l {stats['v_loss']:.1f}", flush=True)

        # 周期性贪心评估（固定种子保证各次评估可比；fast 批量评估与串行逐局等价）
        if it % args.eval_every == 0 or it == total_iters:
            eval_fn = run_greedy_episodes_fast if args.env == "fast" else run_greedy_episodes
            ev = eval_fn(model, args.eval_episodes, seed=12345, device=device)
            metric = ev["board_sum"][0] + 250.0 * ev["p>=256"][0]  # 综合作业目标选最优
            best_ema_dec = None
            raw_sum, raw_p512 = ev['board_sum'][0], ev['p>=512'][0]
            if emas:
                raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                for d, st in emas.items():
                    model.load_state_dict({k: v.to(raw_state[k].dtype) for k, v in st.items()})
                    ev_d = eval_fn(model, args.eval_episodes, seed=12345, device=device)
                    m_d = ev_d["board_sum"][0] + 250.0 * ev_d["p>=256"][0]
                    if m_d > metric:
                        ev, metric, best_ema_dec = ev_d, m_d, d
                # 无论哪个 EMA 更优，训练都必须从原始权重继续
                model.load_state_dict(raw_state)
            eval_writer.writerow([it, total_steps,
                                  f"{ev['score'][0]:.1f}", f"{ev['score'][1]:.1f}",
                                  f"{ev['max_tile'][0]:.1f}",
                                  f"{ev['board_sum'][0]:.1f}", f"{ev['board_sum'][1]:.1f}",
                                  f"{ev['length'][0]:.1f}",
                                  f"{ev['p>=128'][0]:.3f}", f"{ev['p>=256'][0]:.3f}",
                                  f"{ev['p>=512'][0]:.3f}", f"{ev['p>=1024'][0]:.3f}",
                                  f"{raw_sum:.1f}", f"{raw_p512:.3f}"])
            eval_csv.flush()
            print(f"  [eval it {it}] score {ev['score'][0]:.0f}±{ev['score'][1]:.0f} | "
                  f"max_tile {ev['max_tile'][0]:.0f} | sum {ev['board_sum'][0]:.0f}±{ev['board_sum'][1]:.0f} | "
                  f"p128 {ev['p>=128'][0]:.2f} | p256 {ev['p>=256'][0]:.2f} | "
                  f"p512 {ev['p>=512'][0]:.2f}", flush=True)
            if metric > best_metric:
                best_metric = metric
                if best_ema_dec is not None:
                    keep = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    st = emas[best_ema_dec]
                    model.load_state_dict({k: v.to(keep[k].dtype) for k, v in st.items()})
                    save_ckpt("best.pt", it)
                    model.load_state_dict(keep)
                else:
                    save_ckpt("best.pt", it)
        if it % 200 == 0:
            save_ckpt("last.pt", it)
            train_csv.flush()

    if emas:
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for d, st in emas.items():
            model.load_state_dict({k: v.to(raw[k].dtype) for k, v in st.items()})
            save_ckpt(f"last_ema_{d:g}.pt", total_iters)
        model.load_state_dict(raw)
    save_ckpt("last.pt", total_iters)
    train_csv.close()
    eval_csv.close()
    print(f"训练完成，用时 {(time.time() - t_start) / 60:.1f} 分钟，结果目录: {out_dir}")


if __name__ == "__main__":
    main()
