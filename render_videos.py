"""用训练好的最优配置模型（10 epochs + lr 6.25e-4）跑 3 个随机种子的完整对局，
逐步渲染为 MP4 视频。贪心策略（掩码 argmax），与最终测试口径一致。

输出：ppo2048/videos/seed_<N>.mp4
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch

from game2048_env import Game2048Env3x3, ACTION_MEANING
from ppo import ActorCritic

BASE = Path(__file__).parent
OUT = BASE / 'videos'
CN_FONT = ['AR PL UKai CN', 'AR PL SungtiL GB', 'WenQuanYi Zen Hei', 'DejaVu Sans']

ARROWS = {0: '←', 1: '→', 2: '↑', 3: '↓'}
TILE_COLORS = {
    0: ('#ebebeb', 'none'), 2: ('#efe6da', '#776e65'), 4: ('#ede0c8', '#776e65'),
    8: ('#f2b179', 'white'), 16: ('#f59563', 'white'), 32: ('#f67c5f', 'white'),
    64: ('#f65e3b', 'white'), 128: ('#edcf72', 'white'), 256: ('#edcc61', 'white'),
    512: ('#edc850', 'white'), 1024: ('#edc53f', 'white'), 2048: ('#edc22e', 'white'),
}


@torch.no_grad()
def play(model, seed, device='cpu'):
    env = Game2048Env3x3()
    obs, info = env.reset(seed=seed)
    frames = [dict(board=[row[:] for row in info['board']], action=None, gain=0.0,
                   score=0, board_sum=info['board_sum'], max_tile=info['max_tile'])]
    done, steps = False, 0
    while not done:
        mask = env.action_mask()
        obs_t = torch.from_numpy(obs[None]).to(device)
        mask_t = torch.tensor(mask)[None].to(device)
        logits = model.policy_logits(obs_t, mask_t).masked_fill(~mask_t, -1e8)
        action = int(logits.argmax(1))
        _, gain, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc
        frames.append(dict(board=[row[:] for row in info['board']], action=action,
                           gain=float(gain), score=info['score'],
                           board_sum=info['board_sum'], max_tile=info['max_tile']))
        obs = env._encode()
    return frames


def render_video(model, seed, fps=8):
    frames = play(model, seed)
    n = len(frames)
    plt.rcParams['font.sans-serif'] = CN_FONT
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(4.6, 5.2), dpi=120)
    ax = fig.add_axes([0.06, 0.30, 0.88, 0.56])
    ax.set_xlim(-.04, 3.04)
    ax.set_ylim(-.04, 3.04)
    ax.set_xticks([])
    ax.set_yticks([])
    rects = [[ax.add_patch(plt.Rectangle((c, 2 - r), 0.965, 0.965, fc='#ebebeb', ec='white', lw=2))
              for c in range(3)] for r in range(3)]
    texts = [[ax.text(c + 0.48, 2 - r + 0.48, '', ha='center', va='center', fontsize=15,
                      fontweight='bold') for c in range(3)] for r in range(3)]
    title = fig.text(0.5, 0.965, '', ha='center', va='top', fontsize=12)
    action_text = fig.text(0.5, 0.895, '', ha='center', va='top', fontsize=11, color='#a0522d')
    stats = fig.text(0.5, 0.16, '', ha='center', va='center', fontsize=11)
    note = fig.text(0.5, 0.05, '', ha='center', va='center', fontsize=10, color='#666666')

    def draw(i):
        f = frames[i]
        for r in range(3):
            for c in range(3):
                v = f['board'][r][c]
                fc, tc = TILE_COLORS.get(v, ('#edc22e', 'white'))
                rects[r][c].set_fc(fc)
                texts[r][c].set_text(str(v) if v else '')
                texts[r][c].set_color(tc if v else 'none')
        ended = i == n - 1
        title.set_text(f'种子 {seed} · 第 {i} / {n - 1} 步' + ('（终局）' if ended else ''))
        if f['action'] is None or ended:
            action_text.set_text('')
        else:
            gain = f'，合并 +{f["gain"]:g}' if f['gain'] else ''
            action_text.set_text(f'动作：{ARROWS[f["action"]]} {ACTION_MEANING[f["action"]]}{gain}')
        stats.set_text(f'得分 {f["score"]:g} · 棋盘和 {f["board_sum"]:g} · 最大块 {f["max_tile"]:g}')
        note.set_text('PPO 三阶段调度 + EMA · 贪心策略' if not ended
                      else 'PPO 三阶段调度 + EMA · 贪心策略 · 终局')
        return []

    anim = FuncAnimation(fig, draw, frames=n, blit=False)
    writer = FFMpegWriter(fps=fps, metadata=dict(title=f'2048 3x3 seed {seed}'))
    out = OUT / f'seed_{seed}.mp4'
    anim.save(str(out), writer=writer)
    plt.close(fig)
    last = frames[-1]
    print(f'{out.name}: {n - 1} 步, 终局棋盘和 {last["board_sum"]}, '
          f'最大块 {last["max_tile"]}, 得分 {last["score"]}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, default=BASE / 'results/sensitivity/epochs10_lr625e4_s2/best.pt')
    p.add_argument('--seeds', nargs='+', type=int, default=[860001, 860002, 860003])
    p.add_argument('--fps', type=int, default=8)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = ActorCritic(hidden=ckpt['args']['hidden'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"模型: {args.ckpt}（第 {ckpt['it']} 迭代）")
    for seed in args.seeds:
        render_video(model, seed, fps=args.fps)


if __name__ == '__main__':
    main()
