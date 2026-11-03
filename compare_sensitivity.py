"""超参数敏感性对比：加载各配置 best.pt，在 1000 局独立测试（种子 70000+）上贪心评估。

每个配置 = 单变量改动（OFAT），3 个训练种子；与基线（主实验配置）逐项对比。
输出 results/sensitivity/ 下的 comparison.json、test_episodes.csv、report.md 与两张图。
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

plt.rcParams.update({
    'font.sans-serif': ['AR PL UKai CN', 'AR PL SungtiL GB', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})

DISPLAY = [
    ('baseline', '基线（γ.995, λ.95, ε.2, 6ep, lr2.5e-4）'),
    ('gamma099', 'γ = 0.99'),
    ('gamma0999', 'γ = 0.999'),
    ('lam09', 'λ = 0.9'),
    ('lam10', 'λ = 1.0'),
    ('clip01', 'ε = 0.1'),
    ('clip03', 'ε = 0.3'),
    ('epochs3', '3 epochs'),
    ('epochs10', '10 epochs'),
    ('epochs10_lr625e4', '最优配置（10 epochs + lr 6.25e-4）'),
    ('combo_lr125e3', '最优配置 + lr 1.25e-3'),
    ('combo_epochs15', '最优配置 + 15 epochs'),
    ('lr1e4', '学习率 1e-4'),
    ('lr625e4', '学习率 6.25e-4'),
    ('ent005', '熵系数减半（.005→.0005）'),
    ('rew005', '奖励缩放 0.05'),
    ('nomask', '无动作掩码'),
]
METRICS = ('score', 'board_sum', 'p>=256', 'p>=512')


def evaluate_run(run_dir, episodes, test_seed):
    ckpt = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    model = ActorCritic(hidden=ckpt['args']['hidden'])
    model.load_state_dict(ckpt['model'])
    _, rows = evaluate(model, 'ppo', episodes, test_seed)
    best_steps = ckpt['it'] * 8192
    return rows, best_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=BASE / 'results/sensitivity')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--test-seed', type=int, default=70000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2, 3, 4, 5, 6],
                        help='参与聚合的训练种子；缺失的运行自动跳过')
    args = parser.parse_args()
    torch.set_num_threads(1)

    rows_all, per_run = [], {}
    for config, label in DISPLAY:
        for seed in args.seeds:
            run_dir = args.root / f'{config}_s{seed}'
            if not (run_dir / 'best.pt').exists():
                print(f'缺少 {run_dir}，跳过（训练未完成或失败）', flush=True)
                continue
            rows, best_steps = evaluate_run(run_dir, args.episodes, args.test_seed)
            done = json.loads((run_dir / 'done.json').read_text()) if (run_dir / 'done.json').exists() else {}
            per_run[(config, seed)] = dict(rows=rows, best_steps=best_steps, train_seconds=done.get('seconds'))
            rows_all += [dict(config=config, training_seed=seed, **r) for r in rows]
            bsum = np.mean([r['board_sum'] for r in rows])
            p256 = np.mean([r['max_tile'] >= 256 for r in rows])
            print(f'{config} s{seed}: sum={bsum:.1f} p256={p256:.1%} '
                  f'(best@{best_steps/1e6:.1f}M, {done.get("seconds", 0)/60:.0f}min)', flush=True)

    aggregate = {}
    for config, label in DISPLAY:
        runs = [v for (c, s), v in per_run.items() if c == config]
        if not runs:
            continue
        entry = {}
        for key in METRICS:
            if key.startswith('p>='):
                values = [np.mean([r['max_tile'] >= int(key[3:]) for r in v['rows']]) for v in runs]
            else:
                values = [np.mean([r[key] for r in v['rows']]) for v in runs]
            entry[key] = dict(mean=float(np.mean(values)), seed_std=float(np.std(values, ddof=1)),
                              seeds=[float(v) for v in values])
        entry['best_steps'] = [v['best_steps'] for v in runs]
        entry['train_minutes'] = float(np.mean([v['train_seconds'] or 0 for v in runs]) / 60)
        aggregate[config] = entry

    baseline = aggregate.get('baseline')
    if baseline:
        for config, _ in DISPLAY:
            if config == 'baseline' or config not in aggregate:
                continue
            for key in METRICS:
                aggregate[config][f'{key}_delta'] = aggregate[config][key]['mean'] - baseline[key]['mean']

    result = dict(episodes_per_model=args.episodes, test_seed_start=args.test_seed,
                  training_budget=24576000, training_seeds=[2, 3, 4],
                  validation_seed=12345, selection_metric='board_sum + 250*P(>=256)',
                  aggregate=aggregate)
    (args.root / 'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with (args.root / 'test_episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_all[0]))
        writer.writeheader()
        writer.writerows(rows_all)
    plot_curves(args.root)
    plot_bars(args.root, aggregate)
    write_report(args.root, result)


def plot_curves(root):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    panels = [('board_sum_mean', '验证棋盘和'), ('p>=256', '验证 P(≥256)'), ('p>=512', '验证 P(≥512)')]
    cmap = plt.get_cmap('tab20')
    for i, (config, label) in enumerate(DISPLAY):
        logs = []
        for seed in (2, 3, 4, 5, 6):
            path = root / f'{config}_s{seed}/eval_log.csv'
            if not path.exists():
                continue
            with path.open() as f:
                logs.append(list(csv.DictReader(f)))
        if not logs:
            continue
        x = np.array([int(r['steps']) for r in logs[0]]) / 1e6
        for ax, (key, title) in zip(axes, panels):
            values = np.array([[float(r[key]) for r in run] for run in logs])
            mean = values.mean(0)
            style = dict(color='#3977ad', lw=2.4) if config == 'baseline' else \
                dict(color=cmap(i % 20), lw=1.1, alpha=.85)
            ax.plot(x, mean, label=label, **style)
    for ax, (_, title) in zip(axes, panels):
        ax.set_xlabel('环境步数（百万）')
        ax.set_title(title, fontsize=11)
        if title != '验证棋盘和':
            ax.set_ylim(-.02, 1.02)
    axes[0].legend(fontsize=7.2, ncol=2, loc='lower right')
    fig.suptitle('验证集学习曲线：每配置 3 种子均值（24.576M 步，每 204,800 步评估 50 局）', fontsize=12)
    fig.tight_layout()
    fig.savefig(root / 'sensitivity_curves.png', dpi=160)
    plt.close(fig)


def plot_bars(root, aggregate):
    present = [(c, l) for c, l in DISPLAY if c in aggregate]
    present.sort(key=lambda pair: aggregate[pair[0]]['board_sum']['mean'])
    labels = [l for _, l in present]
    bsum = [aggregate[c]['board_sum']['mean'] for c, _ in present]
    bstd = [aggregate[c]['board_sum']['seed_std'] for c, _ in present]
    p512 = [aggregate[c]['p>=512']['mean'] * 100 for c, _ in present]
    colors = ['#3977ad' if c == 'baseline' else '#b7c3cf' for c, _ in present]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    bars = ax.barh(labels, bsum, xerr=bstd, capsize=3, color=colors)
    ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=8.5)
    ax.set_xlabel('测试棋盘和（3 种子均值，误差线=种子标准差）')
    ax.set_title('最终棋盘和（1000 局/模型）', fontsize=11)
    ax = axes[1]
    bars = ax.barh(labels, p512, color=colors)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=8.5)
    ax.set_xlabel('P(≥512)（%）')
    ax.set_title('达到 512 的比例', fontsize=11)
    fig.suptitle('PPO 超参数敏感性：单变量改动 × 完整预算 24.576M 步 × 3 训练种子', fontsize=12)
    fig.tight_layout()
    fig.savefig(root / 'sensitivity_bars.png', dpi=160)
    plt.close(fig)


def write_report(root, result):
    a = result['aggregate']
    base = a['baseline']
    lines = ['# PPO 超参数敏感性实验', '',
             '单变量设计（OFAT）：每个配置只改一个超参数，其余与主实验完全一致；全部从头训练'
             f" {result['training_budget']:,} 环境步（完整预算），训练种子 2、3、4。",
             '验证与选模规则与主实验相同（每 204,800 步在固定 50 局上评估，棋盘和 + 250×P(≥256) 选 best）。',
             f"最终测试：每模型 {result['episodes_per_model']} 局，种子 {result['test_seed_start']} 起，贪心执行；测试种子未参与任何选择。", '',
             '| 配置 | 棋盘和 | Δ vs 基线 | P(≥256) | P(≥512) | 平均训练用时 |',
             '|---|---:|---:|---:|---:|---:|']
    for config, label in DISPLAY:
        if config not in a:
            continue
        e = a[config]
        delta = '' if config == 'baseline' else f"{e['board_sum_delta']:+.1f}"
        lines.append(f"| {label} | {e['board_sum']['mean']:.1f} ± {e['board_sum']['seed_std']:.1f} | {delta} | "
                     f"{e['p>=256']['mean']:.1%} | {e['p>=512']['mean']:.1%} | {e['train_minutes']:.0f} 分钟 |")
    lines += ['', '± 为 3 个训练种子间的标准差；Δ 为与基线（同预算重跑）的棋盘和均值差。', '',
              '## 相对基线的解读（自动生成，供人工核对）', '']
    noise = base['board_sum']['seed_std']
    labels = dict(DISPLAY)
    others = sorted([c for c in a if c != 'baseline'], key=lambda c: -a[c]['board_sum_delta'])
    above = [c for c in others if a[c]['board_sum_delta'] > 0]
    below = [c for c in reversed(others) if a[c]['board_sum_delta'] < 0]
    if above:
        lines.append(f"- 高于基线最多：{'、'.join(labels[c] for c in above[:3])}。")
    if below:
        lines.append(f"- 低于基线最多：{'、'.join(labels[c] for c in below[:3])}。")
    lines.append(f"- 基线自身的种子标准差为 ±{noise:.1f}，棋盘和差异在 ±{noise:.1f} 以内的配置视为对种子噪声不敏感。")
    if 'nomask' in a:
        lines.append(f"- 无动作掩码：棋盘和 {a['nomask']['board_sum']['mean']:.1f}（Δ{a['nomask']['board_sum_delta']:+.1f}），"
                     f"P(≥256) {a['nomask']['p>=256']['mean']:.1%}——短预算消融中最大的负面效应在完整预算下的表现。")
    lines += ['', '## 口径与限制', '',
              '- 单变量 OFAT：只考察一维敏感性，不考察超参数之间的交互。',
              '- 每配置 3 个训练种子；± 是种子间标准差，不是置信区间；差异小于基线种子标准差的结论不构成显著性的证据。',
              '- 所有配置共享同一验证集、选模规则与测试种子；结论限定于本自写 3×3 环境与合并得分奖励。',
              '- 训练耗时为单配置 3 种子平均墙钟（8 并发调度下），不代表严格同硬件速度排名。', '']
    (root / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
