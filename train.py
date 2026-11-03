"""PPO 训练脚本：同步向量化环境采集 -> GAE -> PPO-clip 更新。

用法示例：
    python train.py --run-name main --iters 3000
    python train.py --run-name ablation_low_ent --iters 400 --ent-coef 0.002 --ent-final 0.0002

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
from ppo import ActorCritic, compute_gae, ppo_update, run_greedy_episodes


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
    p.add_argument("--no-mask", action="store_true", help="消融：不使用动作掩码")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
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


def main():
    args = parse_args()
    out_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)
    envs = [Game2048Env3x3() for _ in range(args.n_envs)]
    obs_list, mask_list = [], []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=args.seed * 10_000 + i)
        obs_list.append(obs)
        mask_list.append(env.action_mask())

    model = ActorCritic(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = RolloutBuffer(args.n_envs, args.steps, envs[0].observation_space.shape[0], device)

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
                          "p>=128", "p>=256", "p>=512", "p>=1024"])

    def save_ckpt(name, it):
        torch.save({"it": it, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "args": vars(args)},
                   os.path.join(out_dir, name))

    print(f"[{args.run_name}] device={device}, 每迭代 {args.n_envs * args.steps} 步, "
          f"共 {args.iters} 迭代 (≈{args.iters * args.n_envs * args.steps / 1e6:.1f}M 步)")
    t_start = time.time()
    for it in range(1, args.iters + 1):
        # lr 与熵系数线性退火
        frac = (it - 1) / max(args.iters - 1, 1)
        lr_now = args.lr * (1.0 - (1.0 - args.lr_final_ratio) * frac)
        ent_now = args.ent_coef + (args.ent_final - args.ent_coef) * frac
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        model.eval()
        with torch.no_grad():
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

        # bootstrap 末端价值
        with torch.no_grad():
            obs_batch = torch.from_numpy(np.stack(obs_list)).to(device)
            last_values = model.value(obs_batch).cpu().numpy()
        adv, ret = compute_gae(buffer.rew, buffer.val.cpu().numpy(), buffer.done,
                               last_values, args.gamma, args.lam)

        model.train()
        stats = ppo_update(model, optimizer, buffer.flat(adv, ret), args.clip,
                           args.epochs, args.minibatch_size, args.vf_coef, ent_now)
        total_steps = it * args.n_envs * args.steps

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

        # 周期性贪心评估（固定种子保证各次评估可比）
        if it % args.eval_every == 0 or it == args.iters:
            ev = run_greedy_episodes(model, args.eval_episodes, seed=12345, device=device)
            metric = ev["board_sum"][0] + 250.0 * ev["p>=256"][0]  # 综合作业目标选最优
            eval_writer.writerow([it, total_steps,
                                  f"{ev['score'][0]:.1f}", f"{ev['score'][1]:.1f}",
                                  f"{ev['max_tile'][0]:.1f}",
                                  f"{ev['board_sum'][0]:.1f}", f"{ev['board_sum'][1]:.1f}",
                                  f"{ev['length'][0]:.1f}",
                                  f"{ev['p>=128'][0]:.3f}", f"{ev['p>=256'][0]:.3f}",
                                  f"{ev['p>=512'][0]:.3f}", f"{ev['p>=1024'][0]:.3f}"])
            eval_csv.flush()
            print(f"  [eval it {it}] score {ev['score'][0]:.0f}±{ev['score'][1]:.0f} | "
                  f"max_tile {ev['max_tile'][0]:.0f} | sum {ev['board_sum'][0]:.0f}±{ev['board_sum'][1]:.0f} | "
                  f"p128 {ev['p>=128'][0]:.2f} | p256 {ev['p>=256'][0]:.2f} | "
                  f"p512 {ev['p>=512'][0]:.2f}", flush=True)
            if metric > best_metric:
                best_metric = metric
                save_ckpt("best.pt", it)
        if it % 200 == 0:
            save_ckpt("last.pt", it)
            train_csv.flush()

    save_ckpt("last.pt", args.iters)
    train_csv.close()
    eval_csv.close()
    print(f"训练完成，用时 {(time.time() - t_start) / 60:.1f} 分钟，结果目录: {out_dir}")


if __name__ == "__main__":
    main()
