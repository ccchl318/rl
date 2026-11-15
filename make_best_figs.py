"""生成当前最优配置的报告图表（第二阶段优化）。

输出 figures/best/ 下的 PNG：
  fig1_lr_schedule.png      四代调度的学习率设计对比
  fig2_training_curves.png  训练过程：棋盘和与 P(≥512) 随迭代的变化
  fig3_final_results.png    1000 局终测的四项指标对比
  fig4_tile_distribution.png 最大块终局分布
  fig5_score_distribution.png 期望得分与对局长度分布
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
OUT = BASE / "figures" / "best"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UKai CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

C = {"base": "#8a8f98", "two": "#5aa9e6", "three": "#2a9d8f", "combo": "#e76f51"}

# ---------------- 数据装载 ----------------

def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def eval_curve(paths):
    """多曲线 → 按迭代平均（返回 iter, board_sum, p512）。"""
    acc = {}
    for p in paths:
        for r in read_csv(p):
            it = int(r["iter"])
            acc.setdefault(it, []).append((float(r["board_sum_mean"]), float(r["p>=512"])))
    its = sorted(acc)
    bs = np.array([np.mean([x[0] for x in acc[i]]) for i in its])
    p5 = np.array([np.mean([x[1] for x in acc[i]]) for i in its])
    return np.array(its), bs, p5


def episodes(paths, budget_filter=None):
    tiles, sums, scores, lens = [], [], [], []
    for p in paths:
        for r in read_csv(p):
            if budget_filter and r.get("budget") != budget_filter:
                continue
            tiles.append(int(r["max_tile"]))
            sums.append(float(r["board_sum"]))
            scores.append(float(r["score"]))
            lens.append(int(r["length"]))
    return np.array(tiles), np.array(sums), np.array(scores), np.array(lens)


R = BASE / "results"
CFG = {
    "标准 2× 基线": dict(
        curves=[R / f"budget_extension/epochs10_lr625e4_s{s}/eval_log.csv" for s in (2, 3, 4)],
        eps=[R / "budget_extension/test_episodes.csv"], budget="49.152M", color=C["base"]),
    "两阶段（+精修 6e-5）": dict(
        curves=[R / f"ext2x_s{s}_eval_log.csv" for s in (2, 3, 4)],
        eps=[R / f"ext2x_eval/ext2x_s{s}_test_episodes.csv" for s in (2, 3, 4)], color=C["two"]),
    "三阶段（+超精修 2e-5）": dict(
        curves=[R / f"final_v2/ultrafine_s{s}_eval_log.csv" for s in (2, 3, 4)],
        eps=[R / f"final_v2/ultrafine_s{s}_test_episodes.csv" for s in (2, 3, 4)], color=C["three"]),
    "组合（四阶段 + 双 EMA）": dict(
        curves=[R / f"final_v2/combo_s{s}_eval_log.csv" for s in (2, 3, 4)],
        eps=[R / f"final_v2/combo_s{s}_test_episodes.csv" for s in (2, 3, 4)], color=C["combo"]),
}


def rolling(x, w=7):
    if len(x) < w:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


# ---------------- 图 1：学习率调度 ----------------

def fig_lr_schedule():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    lr0, ratio = 6.25e-4, 0.1

    n = 12000
    its = np.arange(1, n + 1)
    ax.plot(its, lr0 * (1 - (1 - ratio) * (its - 1) / (n - 1)),
            color=C["base"], lw=2.2, label="标准 2× 基线（单段退火，12000 迭代）")

    n = 12000
    its = np.arange(1, n + 1)
    lr = np.where(its <= 6000, lr0 * (1 - (1 - ratio) * (its - 1) / 5999), 6e-5)
    ax.plot(its, lr, color=C["two"], lw=2.2, label="两阶段（+6000 @ 6e-5）")

    n = 18000
    its = np.arange(1, n + 1)
    lr = np.where(its <= 6000, lr0 * (1 - (1 - ratio) * (its - 1) / 5999),
                  np.where(its <= 12000, 6e-5, 2e-5))
    ax.plot(its, lr, color=C["three"], lw=2.4, label="三阶段（+6000 @ 2e-5）★ 推荐")

    n = 21000
    its = np.arange(1, n + 1)
    lr = np.where(its <= 6000, lr0 * (1 - (1 - ratio) * (its - 1) / 5999),
                  np.where(its <= 12000, 6e-5,
                           np.where(its <= 18000, 2e-5, 5e-6)))
    ax.plot(its, lr, color=C["combo"], lw=1.8, ls="--", label="组合（+3000 @ 5e-6）")

    ax.set_yscale("log")
    ax.set_xlabel("训练迭代")
    ax.set_ylabel("学习率（对数轴）")
    ax.set_title("图 1  学习率调度设计：越到后期，步长越要小")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8.5, loc="upper right")
    for x, txt in ((6000, "阶段一结束\nlr→6.25e-5"), (12000, "阶段二结束\nlr→2e-5")):
        ax.axvline(x, color="#bbb", ls=":", lw=1)
        ax.text(x + 250, 7e-6, txt, fontsize=7.5, color="#666", va="bottom")
    fig.savefig(OUT / "fig1_lr_schedule.png")
    plt.close(fig)


# ---------------- 图 2：训练曲线 ----------------

def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for name, cfg in CFG.items():
        its, bs, p5 = eval_curve(cfg["curves"])
        axes[0].plot(its, rolling(bs), color=cfg["color"], lw=1.9, label=name)
        axes[1].plot(its, rolling(p5) * 100, color=cfg["color"], lw=1.9, label=name)
    axes[0].set_xlabel("训练迭代"); axes[0].set_ylabel("棋盘和（贪心评估，50 局）")
    axes[0].set_title("(a) 棋盘和随训练的变化")
    axes[1].set_xlabel("训练迭代"); axes[1].set_ylabel("P(最大块 ≥ 512)  %")
    axes[1].set_title("(b) 合成 512 的概率随训练的变化")
    for ax in axes:
        ax.grid(alpha=0.25, ls=":")
        ax.legend(fontsize=8)
    fig.suptitle("图 2  训练过程：后期低学习率精修带来持续提升", y=1.02)
    fig.savefig(OUT / "fig2_training_curves.png")
    plt.close(fig)


# ---------------- 图 3：最终结果 ----------------

def fig_final_results():
    names, data = [], []
    for name, cfg in CFG.items():
        tiles, sums, scores, lens = episodes(cfg["eps"], cfg.get("budget"))
        names.append(name); data.append((tiles, sums, scores))
    metrics = [
        ("棋盘和", lambda d: d[1].mean(), lambda d: d[1].std()),
        ("P(≥256) %", lambda d: (d[0] >= 256).mean() * 100, None),
        ("P(≥512) %", lambda d: (d[0] >= 512).mean() * 100, None),
        ("P(≥1024) %（千局占比）", lambda d: (d[0] >= 1024).mean() * 100, None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9))
    for ax, (title, f, err) in zip(axes, metrics):
        vals = [f(d) for d in data]
        cols = [cfg["color"] for cfg in CFG.values()]
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.68)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(["基线", "两阶段", "三阶段", "组合"], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, ls=":", axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8.5)
        if err:
            ax.set_ylim(0, max(vals) * 1.18)
        elif max(vals) < 5:
            ax.set_ylim(0, max(max(vals) * 2.2, 1.0))   # 极小比例面板自适应
        else:
            ax.set_ylim(0, 100)
    fig.suptitle("图 3  1000 局终测结果（种子 70000–70999，贪心执行）", y=1.04)
    fig.savefig(OUT / "fig3_final_results.png")
    plt.close(fig)


# ---------------- 图 4：最大块分布 ----------------

def fig_tile_distribution():
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    levels = [64, 128, 256, 512, 1024, 2048]
    width = 0.38
    for i, (label, key) in enumerate((("标准 2× 基线", "标准 2× 基线"), ("三阶段（推荐）", "三阶段（+超精修 2e-5）"))):
        cfg = CFG[key]
        tiles, *_ = episodes(cfg["eps"], cfg.get("budget"))
        freqs = [np.mean(tiles >= L) * 100 for L in levels]
        xs = np.arange(len(levels)) + (i - 0.5) * width
        bars = ax.bar(xs, freqs, width=width, color=cfg["color"], label=label)
        for b, v in zip(bars, freqs):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"≥{L}" for L in levels])
    ax.set_ylabel("达到该块的比例  %")
    ax.set_xlabel("终局最大块")
    ax.set_ylim(0, 108)
    ax.set_title("图 4  终局最大块分布：长局（≥512）比例从 53% 提升到 67%")
    ax.grid(alpha=0.25, ls=":", axis="y")
    ax.legend(fontsize=9)
    fig.savefig(OUT / "fig4_tile_distribution.png")
    plt.close(fig)


# ---------------- 图 5：得分与长度分布 ----------------

def fig_score_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for key, label in (("标准 2× 基线", "标准 2× 基线"), ("三阶段（+超精修 2e-5）", "三阶段（推荐）")):
        cfg = CFG[key]
        tiles, sums, scores, lens = episodes(cfg["eps"], cfg.get("budget"))
        axes[0].hist(scores, bins=40, alpha=0.55, color=cfg["color"], label=f"{label}（均值 {scores.mean():.0f}）", density=True)
        axes[1].hist(lens, bins=40, alpha=0.55, color=cfg["color"], label=f"{label}（均值 {lens.mean():.0f} 步）", density=True)
    axes[0].set_xlabel("单局得分"); axes[0].set_ylabel("概率密度")
    axes[0].set_title("(a) 得分分布")
    axes[1].set_xlabel("对局长度（步）"); axes[1].set_ylabel("概率密度")
    axes[1].set_title("(b) 对局长度分布")
    for ax in axes:
        ax.grid(alpha=0.25, ls=":", axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("图 5  对局统计分布（各 1000 局）", y=1.02)
    fig.savefig(OUT / "fig5_score_distribution.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_lr_schedule(); print("fig1 ok")
    fig_training_curves(); print("fig2 ok")
    fig_final_results(); print("fig3 ok")
    fig_tile_distribution(); print("fig4 ok")
    fig_score_distribution(); print("fig5 ok")
