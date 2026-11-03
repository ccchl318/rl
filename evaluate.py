"""最终评估脚本：加载训练好的模型（或随机策略基线），跑若干局贪心对局，
输出统计表、训练曲线图、最大块分布直方图以及最优对局棋谱。

用法：
    python evaluate.py --ckpt results/main/best.pt --episodes 200
    python evaluate.py --random --episodes 200        # 随机策略基线
"""

import argparse
import csv
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from game2048_env import Game2048Env3x3, ACTION_MEANING
from ppo import ActorCritic, run_greedy_episodes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--random", action="store_true", help="评估随机(合法动作均匀)策略作为基线")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=555)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tag", type=str, default=None, help="输出文件名后缀")
    return p.parse_args()


@torch.no_grad()
def run_random_episodes(n_episodes: int, seed: int):
    """随机基线：在合法动作中均匀采样。"""
    env = Game2048Env3x3()
    rng = np.random.default_rng(0)
    scores, max_tiles, sums, lens = [], [], [], []
    for i in range(n_episodes):
        _, info = env.reset(seed=seed + i)
        done, length = False, 0
        while not done:
            legal = np.flatnonzero(env.action_mask())
            _, _, term, trunc, info = env.step(int(legal[rng.integers(len(legal))]))
            done = term or trunc
            length += 1
        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        sums.append(info["board_sum"])
        lens.append(length)
    return {"score": scores, "max_tile": max_tiles, "board_sum": sums, "length": lens}


def summarize(raw: dict, episodes: int) -> dict:
    stats = {k: (float(np.mean(v)), float(np.std(v))) for k, v in raw.items()}
    for t in (128, 256, 512, 1024):
        stats[f"p>={t}"] = (float(np.mean([m >= t for m in raw["max_tile"]])), 0.0)
    return stats


def plot_max_tile_hist(counter: Counter, episodes: int, out_path: str):
    tiles = sorted(counter)
    counts = [counter[t] for t in tiles]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([str(t) for t in tiles], counts, color="tab:blue")
    for x, c in zip(range(len(tiles)), counts):
        ax.text(x, c + max(counts) * 0.012, f"{c / episodes:.0%}", ha="center", fontsize=9)
    ax.set_xlabel("max tile in episode")
    ax.set_ylabel("# episodes")
    ax.set_title(f"Distribution of max tile over {episodes} greedy episodes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(run_dir: str, out_path: str):
    """由训练日志画学习曲线。"""
    def read_csv(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return list(csv.DictReader(f))

    ev = read_csv(os.path.join(run_dir, "eval_log.csv"))
    tr = read_csv(os.path.join(run_dir, "train_log.csv"))
    if ev is None:
        return False
    steps = [float(r["steps"]) / 1e6 for r in ev]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(steps, [float(r["score_mean"]) for r in ev], color="tab:blue", label="mean score")
    ax.fill_between(steps,
                    [float(r["score_mean"]) - float(r["score_std"]) for r in ev],
                    [float(r["score_mean"]) + float(r["score_std"]) for r in ev],
                    alpha=0.2, color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(steps, [float(r["board_sum_mean"]) for r in ev], color="tab:red",
             label="mean board sum")
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel("score")
    ax2.set_ylabel("board sum")
    ax.set_title("Greedy eval: score & board sum")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")

    ax = axes[1]
    for t in ("p>=128", "p>=256", "p>=512"):
        ax.plot(steps, [float(r[t]) for r in ev], label=t, marker=".")
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Greedy eval: reach probability")
    ax.legend()

    ax = axes[2]
    if tr:
        ts = [float(r["steps"]) / 1e6 for r in tr]
        ax.plot(ts, [float(r["score_mean"]) for r in tr], label="train score (avg of last 100 eps)")
        ax.plot(ts, [float(r["entropy"]) for r in tr], label="policy entropy")
        ax.set_xlabel("env steps (M)")
        ax.set_title("Training statistics")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.ckpt) if args.ckpt else os.path.join(
        os.path.dirname(__file__), "results")
    tag = args.tag or ("random" if args.random else None)
    suffix = f"_{tag}" if tag else ""

    if args.random:
        raw = run_random_episodes(args.episodes, args.seed)
        best_ep = None
        model_desc = "random legal-action policy (baseline)"
    else:
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        model = ActorCritic(hidden=ckpt["args"]["hidden"]).to(args.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        res = run_greedy_episodes(model, args.episodes, seed=args.seed,
                                  device=args.device, record=True)
        raw = res.pop("raw")
        best_ep = res.pop("best_episode")
        model_desc = f"PPO greedy policy ({os.path.basename(args.ckpt)}, iter {ckpt['it']})"

    stats = summarize(raw, args.episodes)
    tile_counter = Counter(raw["max_tile"])

    lines = []
    lines.append("=" * 62)
    lines.append(f"2048 (3x3) PPO 实验评估  —  policy: {model_desc}")
    lines.append(f"episodes: {args.episodes}, eval seed: {args.seed}")
    lines.append("=" * 62)
    for k in ("score", "board_sum", "max_tile", "length"):
        m, s = stats[k]
        lines.append(f"{k:>12s}: {m:10.1f} ± {s:8.1f}")
    for t in (128, 256, 512, 1024):
        lines.append(f"   P(max tile >= {t:4d}) : {stats[f'p>={t}'][0]:6.1%}")
    lines.append("-" * 62)
    lines.append("max tile 分布：")
    for tile in sorted(tile_counter):
        n = tile_counter[tile]
        bar = "#" * int(round(60 * n / max(args.episodes, 1)))
        lines.append(f"  {tile:5d}: {n:4d} ({n / args.episodes:5.1%}) {bar}")
    report = "\n".join(lines)
    print(report)

    txt_path = os.path.join(out_dir, f"eval_report{suffix}.txt")
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    plot_max_tile_hist(tile_counter, args.episodes,
                       os.path.join(out_dir, f"max_tile_hist{suffix}.png"))

    if args.ckpt:
        ok = plot_training_curves(out_dir, os.path.join(out_dir, f"training_curves{suffix}.png"))
        print(f"训练曲线: {'已保存' if ok else '未找到 eval_log.csv，跳过'}")

    if best_ep and best_ep.get("traj"):
        path = os.path.join(out_dir, f"best_episode{suffix}.txt")
        with open(path, "w") as f:
            f.write(f"最优对局: board_sum={best_ep['board_sum']}, "
                    f"score={best_ep['score']}, max_tile={best_ep['max_tile']}, "
                    f"steps={len(best_ep['traj'])}\n\n")
            for step, (board, action) in enumerate(best_ep["traj"]):
                f.write(f"--- step {step:3d}, action: {ACTION_MEANING[action]} ---\n")
                for row in board:
                    f.write(" ".join(f"{x:5d}" for x in row) + "\n")
                f.write("\n")
        print(f"最优对局棋谱: {path}")
    print(f"评估报告: {txt_path}")


if __name__ == "__main__":
    main()
