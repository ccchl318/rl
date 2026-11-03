"""生成 PPO 实现文档的全部 PNG 图表，输出到脚本所在目录的 figures/。

所有图都从 results/ 下的现有实验数据计算，可重复运行。
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).parent
OUT = BASE / 'figures'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': ['AR PL UKai CN', 'AR PL SungtiL GB', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
C_PPO, C_OTD = '#3977ad', '#c55958'
C_BOX, C_EDGE, C_GRAY = '#eef3f9', '#3977ad', '#666666'


def load_logs():
    # 最优配置（10 epochs + lr 6.25e-4）的 3 个训练种子日志。
    ev, tr = [], []
    for seed in (2, 3, 4):
        run = BASE / 'results/sensitivity' / f'epochs10_lr625e4_s{seed}'
        with (run / 'eval_log.csv').open() as f:
            ev.append(list(csv.DictReader(f)))
        with (run / 'train_log.csv').open() as f:
            tr.append(list(csv.DictReader(f)))
    return ev, tr


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.03',
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)


def arrow(ax, x0, y0, x1, y1, text='', color=C_EDGE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=16, color=color, lw=1.6))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.035, text, ha='center', fontsize=9, color=C_GRAY)


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.2)
    ax.axis('off')
    box(ax, 0.2, 1.9, 2.0, 0.9, '同步向量化采集\n64 环境 × 128 步\n= 8192 步/迭代')
    box(ax, 2.7, 1.9, 1.9, 0.9, 'GAE(λ=.95)\nγ=.995\n末端价值 bootstrap')
    box(ax, 5.1, 1.9, 2.2, 0.9, 'PPO-clip 更新\n6 epochs × minibatch 2048\nε=.2, 优势标准化')
    box(ax, 7.8, 1.9, 1.7, 0.9, 'Adam 更新 θ\nglobal norm ≤ .5')
    box(ax, 5.1, 0.3, 2.2, 0.9, '每 25 迭代验证\n50 局贪心，种子 12345+', fc='#fdf3ec', ec=C_OTD)
    box(ax, 7.8, 0.3, 1.7, 0.9, 'best.pt\n棋盘和+250·P(≥256)', fc='#fdf3ec', ec=C_OTD)
    arrow(ax, 2.2, 2.35, 2.7, 2.35)
    arrow(ax, 4.6, 2.35, 5.1, 2.35)
    arrow(ax, 7.3, 2.35, 7.8, 2.35)
    arrow(ax, 8.65, 1.9, 6.2, 1.2, '下一迭代', color=C_GRAY)
    arrow(ax, 7.3, 0.75, 7.8, 0.75)
    ax.text(5.5, 3.05, '共 3000 迭代 = 24,576,000 环境步；最优配置：学习率 6.25e-4→6.25e-5、'
            '熵系数 .02→.002、10 epochs 线性退火',
            ha='center', fontsize=10, color=C_GRAY)
    fig.savefig(OUT / 'fig1_pipeline.png')
    plt.close(fig)


def tile_color(v):
    if v == 0:
        return '#ebebeb'
    k = min(int(np.log2(v)), 10)
    shades = ['#efe6da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', '#f65e3b',
              '#edcf72', '#edcc61', '#edc850', '#edc53f', '#edc22e']
    return shades[min(k - 1, len(shades) - 1)]


def draw_board(ax, board, x0=0, cell=1.0, fs=13):
    for r in range(3):
        for c in range(3):
            v = board[r][c]
            ax.add_patch(plt.Rectangle((x0 + c * cell, (2 - r) * cell), cell, cell,
                                       fc=tile_color(v), ec='white', lw=2))
            if v:
                ax.text(x0 + c * cell + cell / 2, (2 - r) * cell + cell / 2, str(v),
                        ha='center', va='center', fontsize=fs, fontweight='bold',
                        color='#776e65' if v < 8 else 'white')


def fig2_state_encoding():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), width_ratios=[1, 0.25, 1.5])
    ax = axes[0]
    board = [[2, 0, 4], [128, 2, 0], [0, 64, 8]]
    draw_board(ax, board)
    ax.set_xlim(-.05, 3.05)
    ax.set_ylim(-.3, 3.05)
    ax.axis('off')
    ax.set_title('3×3 棋盘', fontsize=11)
    ax.text(1.5, -.28, '空格=0，块=2 的幂', ha='center', fontsize=9, color=C_GRAY)
    axes[1].axis('off')
    arrow(axes[1], 0, .5, 1, .5, '')
    axes[1].text(.5, .68, 'one-hot', ha='center', fontsize=10, color=C_GRAY)
    axes[1].text(.5, .32, 'log2 分箱', ha='center', fontsize=10, color=C_GRAY)
    ax = axes[2]
    for cell_idx in range(9):
        r, c = divmod(cell_idx, 3)
        v = board[r][c]
        k = min(int(v).bit_length() - 1, 15) if v else 0
        ax.add_patch(plt.Rectangle((c * 5.6, -(r * 3.4)), 5.4, 3.1, fc='#f7f7f7', ec='#cccccc'))
        for b in range(16):
            active = b == k
            ax.add_patch(plt.Rectangle((c * 5.6 + .15 + b * .33, -(r * 3.4) + .3), .3, 2.5,
                                       fc=C_PPO if active else '#e3e9f0', ec='none'))
        ax.text(c * 5.6 + 2.7, -(r * 3.4) - .25, f'格{cell_idx}: bin={k}' + ('（空）' if v == 0 else f'（log2({v})={k}）'),
                ha='center', va='top', fontsize=8, color=C_GRAY)
    ax.set_xlim(-.2, 17)
    ax.set_ylim(-11, 1)
    ax.axis('off')
    ax.set_title('144 维 = 9 格 × 16 bin 的 one-hot', fontsize=11)
    fig.suptitle('状态定义：棋盘 → 每 3×3 格的 log2 值 one-hot 拼接', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_state_encoding.png')
    plt.close(fig)


def fig3_network():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def layer(x, y, w, h, label, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
                                    fc=fc, ec='#888888', lw=1.1))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=9)

    # 输入
    layer(0.2, 2.0, 1.5, 1.2, '输入 obs\n144 维\none-hot', '#f0f0f0')
    # 策略塔
    layer(3.0, 3.6, 1.6, 0.8, 'Linear 256 + ReLU', '#e8f0f8')
    layer(5.1, 3.6, 1.6, 0.8, 'Linear 256 + ReLU', '#e8f0f8')
    layer(7.2, 3.6, 1.6, 0.8, 'Linear 4（logits）', '#dbe9f7')
    layer(9.4, 3.6, 1.9, 0.8, '掩码(-1e8)\n→ softmax → π(a|s)', '#cfe2f3')
    # 价值塔
    layer(3.0, 0.7, 1.6, 0.8, 'Linear 256 + ReLU', '#eef7ee')
    layer(5.1, 0.7, 1.6, 0.8, 'Linear 256 + ReLU', '#eef7ee')
    layer(7.2, 0.7, 1.6, 0.8, 'Linear 1', '#dcefe0')
    layer(9.4, 0.7, 1.9, 0.8, 'V(s) 标量', '#d3ead8')

    def chain(points):
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            arrow(ax, x0, y0, x1, y1)

    chain([(1.7, 2.6), (2.6, 3.95), (3.0, 4.0)])
    chain([(4.6, 4.0), (5.1, 4.0), (6.7, 4.0), (7.2, 4.0), (8.8, 4.0), (9.4, 4.0)])
    chain([(1.7, 2.5), (2.6, 1.1), (3.0, 1.1)])
    chain([(4.6, 1.1), (5.1, 1.1), (6.7, 1.1), (7.2, 1.1), (8.8, 1.1), (9.4, 1.1)])
    ax.text(5.9, 4.75, '策略网络 π（正交初始化：隐层 gain=√2，输出 gain=0.01）',
            ha='center', fontsize=10)
    ax.text(5.9, 0.15, '价值网络 V（同结构，输出 gain=1.0）；两塔独立、不共享参数',
            ha='center', fontsize=10, color=C_GRAY)
    ax.text(5.9, -0.35, '参数量：策略 103,940 + 价值 103,169 = 207,109', ha='center',
            fontsize=10, color=C_GRAY)
    fig.savefig(OUT / 'fig3_network.png')
    plt.close(fig)


def fig4_reward():
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), width_ratios=[1.4, 1])
    ax = axes[0]
    draw_board(ax, [[0, 2, 2], [0, 0, 0], [0, 4, 0]], x0=0)
    draw_board(ax, [[4, 0, 2], [0, 0, 0], [4, 0, 0]], x0=4.6)
    arrow(ax, 3.15, 1.5, 4.55, 1.5)
    ax.text(3.85, 1.85, '←', ha='center', fontsize=15, color=C_OTD)
    ax.text(1.5, 3.25, '移动前', ha='center', fontsize=10)
    ax.text(6.1, 3.25, '合并并生成新块后', ha='center', fontsize=10)
    ax.text(3.85, .55, '本步合并得分 = 4', ha='center', fontsize=10, color=C_OTD)
    ax.set_xlim(-.1, 7.7)
    ax.set_ylim(-.35, 3.6)
    ax.axis('off')
    ax = axes[1]
    ax.axis('off')
    ax.text(0.05, 0.86, '奖励定义', fontsize=12, fontweight='bold')
    ax.text(0.05, 0.62, r'$r_t = 0.25 \times \sum$（本步新合成块的值）', fontsize=13)
    ax.text(0.05, 0.40, r'上例：$r_t = 0.25 \times 4 = 1.0$', fontsize=12)
    ax.text(0.05, 0.20, '无效动作奖励 r = -1（动作掩码下不出现）', fontsize=11)
    ax.text(0.05, 0.02, '回报 $G_t=\\sum_t \\gamma^t r_t$，$\\gamma=0.995$', fontsize=11, color=C_GRAY)
    fig.suptitle('奖励定义：合并得分的固定缩放（缩放不改变最优策略）', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_reward.png')
    plt.close(fig)


def fig5_learning_curves(ev_seeds):
    n = len(ev_seeds)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    panels = [('board_sum_mean', '验证棋盘和（50 局均值）', None),
              ('p>=256', '验证 P(≥256)', (-.02, 1.02)),
              ('p>=512', '验证 P(≥512)', (-.02, 1.02))]
    for ax, (key, title, ylim) in zip(axes, panels):
        values = np.array([[float(r[key]) for r in run] for run in ev_seeds])
        steps = np.array([int(r['steps']) for r in ev_seeds[0]]) / 1e6
        mean, std = values.mean(0), values.std(0, ddof=1)
        ax.plot(steps, mean, color=C_PPO, lw=1.8, label=f'均值（{n} 个训练种子）')
        ax.fill_between(steps, mean - std, mean + std, color=C_PPO, alpha=.16, label='±种子标准差')
        ax.axvline(24.576, color='#bbbbbb', ls=':', lw=1.1)
        ax.set_xlabel('环境步数（百万）')
        ax.set_title(title, fontsize=11)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].legend(fontsize=8)
    fig.suptitle('最优配置验证曲线：10 epochs + lr 6.25e-4，每 204,800 步在同一批 50 局（种子 12345+）上贪心评估', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_learning_curves.png')
    plt.close(fig)


def fig6_train_stats(tr_seeds):
    n = len(tr_seeds)
    it = np.array([int(r['iter']) for r in tr_seeds[0]])

    def stack(key):
        return np.array([[float(r[key]) for r in run] for run in tr_seeds])

    def m(key):
        s = stack(key)
        return s.mean(0)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4))
    ax = axes[0][0]
    ax.plot(it, m('score_mean'), color=C_PPO, lw=1.4, label=f'训练得分（近100局，{n}种子均值）')
    ax.set_xlabel('迭代（×8192 步）')
    ax.set_title('训练局分与棋盘和', fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(it, m('board_sum_mean'), color=C_OTD, lw=1.4, label='棋盘和')
    ax2.spines[['top']].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax = axes[0][1]
    ax.plot(it, m('entropy'), color=C_PPO, lw=1.4, label='策略熵')
    ax.plot(it, m('ent_coef') * 50, color=C_OTD, lw=1.1, ls='--', label='熵系数 ×50')
    ax.set_xlabel('迭代')
    ax.set_title('熵与熵系数退火', fontsize=11)
    ax.legend(fontsize=8)
    ax = axes[1][0]
    ax.semilogy(it, np.maximum(np.abs(m('kl')), 1e-5), color=C_PPO, lw=1.4)
    ax.set_xlabel('迭代')
    ax.set_title('|近似 KL|（old_logp - new_logp，截断于 1e-5）', fontsize=10)
    ax = axes[1][1]
    ax.plot(it, m('clip_frac'), color=C_PPO, lw=1.4, label='clip 比例')
    ax3 = ax.twinx()
    ax3.plot(it, m('v_loss'), color='#999999', lw=1.0, ls='--', label='价值损失')
    ax3.spines[['top']].set_visible(False)
    ax.set_xlabel('迭代')
    ax.set_title('截断比例与价值损失', fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.suptitle(f'训练过程统计（{n} 个训练种子均值）', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_train_stats.png')
    plt.close(fig)


def fig7_test_distribution():
    # 最优配置（10 epochs + lr 6.25e-4）的全部测试逐局数据。
    test_csv = BASE / 'results/sensitivity/test_episodes.csv'
    rows = [r for r in csv.DictReader(open(test_csv))
            if r['config'] == 'epochs10_lr625e4' and r['training_seed'].isdigit()]
    tiles = [int(r['max_tile']) for r in rows]
    bsums = [float(r['board_sum']) for r in rows]
    n_seeds = len(set(r['training_seed'] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    values, counts = np.unique(tiles, return_counts=True)
    bars = ax.bar([str(v) for v in values], counts / len(tiles) * 100, color=C_PPO)
    ax.bar_label(bars, fmt='%.1f%%', fontsize=8, padding=2)
    ax.set_xlabel('单局最大块')
    ax.set_ylabel('比例（%）')
    ax.set_title(f'最大块分布（{n_seeds} 种子 × 1000 局）', fontsize=11)
    ax.set_ylim(0, max(counts) / len(tiles) * 115)
    ax = axes[1]
    ax.hist(bsums, bins=36, color=C_PPO, alpha=.85)
    ax.axvline(float(np.mean(bsums)), color=C_OTD, ls='--', lw=1.4,
               label=f'均值 {np.mean(bsums):.1f}')
    ax.axvline(float(np.median(bsums)), color=C_GRAY, ls=':', lw=1.3,
               label=f'中位数 {np.median(bsums):.1f}')
    ax.set_xlabel('单局棋盘和')
    ax.set_ylabel('局数')
    ax.set_title('棋盘和分布', fontsize=11)
    ax.legend(fontsize=9)
    fig.suptitle(f'最优配置最终测试：贪心策略，种子 {70000} 起，共 {len(tiles)} 局（best 检查点）', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'fig7_test_distribution.png')
    plt.close(fig)
    return len(tiles)


def main():
    ev, tr = load_logs()
    fig1_pipeline()
    fig2_state_encoding()
    fig3_network()
    fig4_reward()
    fig5_learning_curves(ev)
    fig6_train_stats(tr)
    n = fig7_test_distribution()
    print(f'已生成 7 张图到 {OUT}（测试局数 {n}）')


if __name__ == '__main__':
    main()
