"""预算充分性检验：最优配置在 49,152,000 步（2 倍预算）下与 24,576,000 步的同种子对比。

每个种子配对比较，输出 results/budget_extension/ 下的 json、md 报告与逐局数据。
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo import ActorCritic
from train_rl import evaluate

BASE = Path(__file__).parent
SEEDS = (2, 3, 4)

plt.rcParams.update({
    'font.sans-serif': ['AR PL UKai CN', 'AR PL SungtiL GB', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})


@torch.no_grad()
def eval_ckpt(path, episodes, test_seed):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = ActorCritic(hidden=ckpt['args']['hidden'])
    model.load_state_dict(ckpt['model'])
    _, rows = evaluate(model, 'ppo', episodes, test_seed)
    return rows, ckpt['it'] * 8192


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--test-seed', type=int, default=70000)
    args = p.parse_args()
    torch.set_num_threads(1)
    root = BASE / 'results/budget_extension'

    summaries, raw = {}, {}
    for seed in SEEDS:
        for budget, path in (
                ('24.576M', BASE / 'results/sensitivity' / f'epochs10_lr625e4_s{seed}/best.pt'),
                ('49.152M', root / f'epochs10_lr625e4_s{seed}/best.pt')):
            rows, best_steps = eval_ckpt(path, args.episodes, args.test_seed)
            summaries[(budget, seed)] = dict(rows=rows, best_steps=best_steps)
            raw[(budget, seed)] = rows
            bsum = np.mean([r['board_sum'] for r in rows])
            p256 = np.mean([r['max_tile'] >= 256 for r in rows])
            print(f'{budget} s{seed}: sum={bsum:.1f} p256={p256:.1%} (best@{best_steps/1e6:.1f}M)', flush=True)

    table = {}
    for budget in ('24.576M', '49.152M'):
        entry = {}
        for key in ('board_sum', 'score'):
            values = [np.mean([r[key] for r in summaries[(budget, s)]['rows']]) for s in SEEDS]
            entry[key] = dict(mean=float(np.mean(values)), seed_std=float(np.std(values, ddof=1)),
                              seeds=[float(v) for v in values])
        for tile in (256, 512):
            values = [np.mean([r['max_tile'] >= tile for r in summaries[(budget, s)]['rows']]) for s in SEEDS]
            entry[f'p>={tile}'] = dict(mean=float(np.mean(values)), seed_std=float(np.std(values, ddof=1)),
                                       seeds=[float(v) for v in values])
        table[budget] = entry

    rng = np.random.default_rng(20261103)
    idx = rng.integers(args.episodes, size=(5000, args.episodes))
    paired = {}
    for seed in SEEDS:
        for key in ('board_sum', 'p>=256', 'p>=512'):
            def vals(rows):
                return np.array([r['max_tile'] >= int(key[3:]) if key.startswith('p>=') else r[key]
                                 for r in rows], dtype=float)
            diff = vals(raw[('49.152M', seed)]) - vals(raw[('24.576M', seed)])
            boot = diff[idx].mean(1)
            paired[f's{seed}_{key}'] = dict(difference=float(diff.mean()),
                                            bootstrap95=np.quantile(boot, [.025, .975]).tolist())

    rows_flat = [dict(budget=b, training_seed=s, **r)
                 for (b, s), rows in raw.items() for r in rows]
    with (root / 'test_episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_flat[0]))
        writer.writeheader()
        writer.writerows(rows_flat)

    result = dict(test_seed_start=args.test_seed, episodes_per_model=args.episodes,
                  seeds=list(SEEDS), budgets=['24.576M', '49.152M'], table=table, paired=paired)
    (root / 'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))

    b24, b49 = table['24.576M'], table['49.152M']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    for ax, key, title in ((axes[0], 'board_sum', '测试棋盘和'), (axes[1], 'p>=512', 'P(≥512)（%）')):
        x = np.arange(2)
        means = [b24[key]['mean'] * (100 if key != 'board_sum' else 1),
                 b49[key]['mean'] * (100 if key != 'board_sum' else 1)]
        stds = [b24[key]['seed_std'] * (100 if key != 'board_sum' else 1),
                b49[key]['seed_std'] * (100 if key != 'board_sum' else 1)]
        bars = ax.bar(['24.576M', '49.152M'], means, yerr=stds, capsize=6,
                      color=['#b7c3cf', '#3977ad'])
        ax.bar_label(bars, fmt='%.1f', padding=5)
        for s in SEEDS:
            xv = [0, 1]
            yv = [b24[key]['seeds'][s - 2] * (100 if key != 'board_sum' else 1),
                  b49[key]['seeds'][s - 2] * (100 if key != 'board_sum' else 1)]
            ax.plot(xv, yv, '--', color='#999999', lw=.9, alpha=.7)
        ax.set_title(title, fontsize=11)
    fig.suptitle('最优配置预算充分性：2 倍预算同种子对比（灰虚线连接同一训练种子）', fontsize=12)
    fig.tight_layout()
    fig.savefig(root / 'budget_comparison.png', dpi=160)
    plt.close(fig)

    lines = ['# 预算充分性检验：最优配置 × 2 倍预算', '',
             f"同一配置（10 epochs + lr 6.25e-4）、同一批训练种子（{', '.join(map(str, SEEDS))}），"
             f"预算从 24,576,000 步扩展至 49,152,000 步（2 倍）；最终测试每模型 {args.episodes} 局，"
             f"种子 {args.test_seed} 起，best 检查点贪心执行。", '',
             '| 预算 | 棋盘和 | P(≥256) | P(≥512) | best 检查点位置（M 步） |',
             '|---|---:|---:|---:|---|']
    for budget in ('24.576M', '49.152M'):
        steps = [summaries[(budget, s)]['best_steps'] / 1e6 for s in SEEDS]
        e = table[budget]
        lines.append(f"| {budget} | {e['board_sum']['mean']:.1f} ± {e['board_sum']['seed_std']:.1f} | "
                     f"{e['p>=256']['mean']:.1%} | {e['p>=512']['mean']:.1%} | "
                     f"{', '.join(f'{x:.1f}' for x in steps)} |")
    lines += ['', '## 逐种子配对差值（49.152M − 24.576M）', '',
              '同一测试种子上的配对 bootstrap 95% 区间，只描述已训练模型的对局波动。', '']
    for key, entry in paired.items():
        scale = 100 if 'p>=' in key else 1
        unit = ' 个百分点' if scale == 100 else ''
        lines.append(f"- {key}: {entry['difference'] * scale:+.1f}{unit}，"
                     f"95% [{entry['bootstrap95'][0] * scale:+.1f}, {entry['bootstrap95'][1] * scale:+.1f}]。")
    (root / 'report.md').write_text('\n'.join(lines))
    print('已写入 budget_extension/report.md')


if __name__ == '__main__':
    main()
