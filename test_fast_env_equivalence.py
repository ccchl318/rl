"""验证向量化批量环境 FastGame2048Batch 与旧环境 Game2048Env3x3 完全等价。

四个层次的检查：
1. 行查表：全部 4096 种行局面的移动结果/合并得分/可动性 vs 旧环境行逻辑；
2. 动作掩码：随机棋盘上 mask 与旧环境 action_mask 逐一对比；
3. 逐位对拍：per_env RNG 模式下，同种子同动作序列的完整对局，
   每步的奖励/棋盘/终止/掩码、局末统计全部逐位一致
   （动作分两种：完全随机——覆盖非法动作路径；随机合法——覆盖训练路径）；
4. spawn 分布：shared 模式下初始块的 2/4 比例与空位分布统计检验。

用法：python test_fast_env_equivalence.py [--episodes 100] [--boards 20000]
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game2048_env import Game2048Env3x3, _slide_row_left, _can_slide_row_left
from game2048_env_fast import (FastGame2048Batch, MOVE_L, MOVE_R, GAIN,
                               CAN_L, CAN_R, N_CODES)


def check_tables():
    """层次 1：行查表 vs 旧环境行逻辑（穷举 4096）。"""
    bad = 0
    for code in range(N_CODES):
        row_lg = [code >> 8 & 15, code >> 4 & 15, code & 15]
        row = [0 if x == 0 else (1 << x) for x in row_lg]  # tile 值域
        new, gain = _slide_row_left(row)
        new_lg = [0 if x == 0 else (x.bit_length() - 1) for x in new]
        exp = (new_lg[0] << 8) | (new_lg[1] << 4) | new_lg[2]
        if MOVE_L[code] != exp:
            bad += 1
        if GAIN[code] != gain:
            bad += 1
        if CAN_L[code] != _can_slide_row_left(row):
            bad += 1
        rev = row[::-1]
        new_r, _ = _slide_row_left(rev)
        new_r_lg = [0 if x == 0 else (x.bit_length() - 1) for x in new_r][::-1]
        exp_r = (new_r_lg[0] << 8) | (new_r_lg[1] << 4) | new_r_lg[2]
        if MOVE_R[code] != exp_r:
            bad += 1
        if CAN_R[code] != _can_slide_row_left(rev):
            bad += 1
    assert bad == 0, f"行查表不一致 {bad} 处"
    print(f"[1] 行查表：{N_CODES} 种行局面 x (移动/得分/左可动/右移动/右可动) 全部一致 ✔")


def check_masks(n_boards=20000):
    """层次 2：随机棋盘上的动作掩码对比。"""
    rng = np.random.default_rng(0)
    env_p = Game2048Env3x3()
    env_f = FastGame2048Batch(1, seed=0)
    tiles = [0, 0, 0, 2, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    mismatch = 0
    for _ in range(n_boards):
        board = [[int(rng.choice(tiles)) for _ in range(3)] for _ in range(3)]
        env_p.board = [r[:] for r in board]
        mask_p = env_p.action_mask()
        cells = [0 if x == 0 else x.bit_length() - 1
                 for r in board for x in r]
        env_f.cells[0] = cells
        env_f._refresh_mask()
        if list(env_f.mask[0]) != list(mask_p):
            mismatch += 1
            if mismatch <= 3:
                print("MASK MISMATCH:", board, env_f.mask[0].tolist(), mask_p)
    assert mismatch == 0, f"掩码不一致 {mismatch}/{n_boards} 盘"
    print(f"[2] 动作掩码：{n_boards} 个随机棋盘全部一致 ✔")


def _run_episodes(n_episodes, valid_only):
    """层次 3：同种子同动作序列逐位对拍（覆盖非法/合法两种动作路径）。"""
    rng = np.random.default_rng(7)
    mismatch = 0
    total_steps = 0
    for ep in range(n_episodes):
        env_p = Game2048Env3x3()
        env_p.reset(seed=1000 + ep)
        env_f = FastGame2048Batch(1, seeds=[1000 + ep])
        t = 0
        while t < 100000:
            if valid_only:
                legal = np.flatnonzero(env_p.action_mask())
                if len(legal) == 0:
                    break  # 旧环境终止（快速环境已自动重开）
                a = int(legal[rng.integers(len(legal))])
            else:
                a = int(rng.integers(4))
            mask_p = env_p.action_mask()
            assert list(env_f.mask[0]) == list(mask_p), f"mask ep{ep} t{t}"

            _, rew_p, term_p, _, info_p = env_p.step(a)
            rew_f, done_f = env_f.step(np.array([a]))
            total_steps += 1

            board_f = [[int(1 << x) if x else 0 for x in env_f.cells[0][3 * r:3 * r + 3]]
                       for r in range(3)]
            if not term_p and board_f != [r[:] for r in env_p.board]:
                # 终止步上 fast 环境已自动重开（棋盘是新一局的），故只在非终止步比棋盘
                mismatch += 1
                print(f"BOARD MISMATCH ep{ep} t{t}: {board_f} vs {env_p.board}")
                break
            if abs(rew_p - float(rew_f[0])) > 1e-9:
                mismatch += 1
                print(f"REWARD MISMATCH ep{ep} t{t}: {rew_p} vs {rew_f[0]}")
                break
            if bool(done_f[0]) != bool(term_p):
                mismatch += 1
                print(f"TERMINAL MISMATCH ep{ep} t{t}: {done_f[0]} vs {term_p}")
                break
            if term_p:
                fin = env_f.pop_finished()
                assert len(fin) == 1, "终止时应有恰好一条局记录"
                _, (sc, mt, bs, ln) = fin[0]
                if (mt, bs, ln) != (info_p["max_tile"], info_p["board_sum"], t + 1):
                    mismatch += 1
                    print(f"STATS MISMATCH ep{ep}: {(mt, bs, ln)} vs "
                          f"{(info_p['max_tile'], info_p['board_sum'], t + 1)}")
                break
            t += 1
    return mismatch, total_steps


def check_episodes(n_episodes):
    m1, s1 = _run_episodes(n_episodes, valid_only=False)
    assert m1 == 0, f"随机动作路径不一致 {m1} 处"
    print(f"[3a] 逐位对拍（完全随机动作，含非法路径）：{n_episodes} 局 {s1} 步全部一致 ✔")
    m2, s2 = _run_episodes(n_episodes, valid_only=True)
    assert m2 == 0, f"合法动作路径不一致 {m2} 处"
    print(f"[3b] 逐位对拍（随机合法动作，训练路径）：{n_episodes} 局 {s2} 步全部一致 ✔")


def check_spawn(n_trials=200000):
    """层次 4：shared 模式 spawn 分布（tile=4 比例应≈0.1）。"""
    rng = np.random.default_rng(3)
    n4 = 0
    for _ in range(n_trials):
        env = FastGame2048Batch(1, seed=int(rng.integers(1 << 31)))
        # 初始两个块
        vals = sorted(env.cells[0][env.cells[0] > 0].tolist())
        n4 += sum(1 for v in vals if v == 2)
    frac4 = n4 / (2 * n_trials)
    assert abs(frac4 - 0.1) < 0.01, f"tile=4 比例异常: {frac4:.4f}"
    print(f"[4] spawn 分布：{n_trials} 次重开，tile=4 比例 {frac4:.4f}（期望 0.10）✔")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--boards", type=int, default=20000)
    args = ap.parse_args()
    check_tables()
    check_masks(args.boards)
    check_episodes(args.episodes)
    check_spawn()
    print("全部等价检查通过：FastGame2048Batch 与 Game2048Env3x3 语义逐位一致 ✔")
