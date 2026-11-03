"""3x3 版 2048 的 Gymnasium 环境（纯 Python 实现，无需 pygame，适合大规模训练）。

规则与经典 2048 一致：
- 棋盘 3x3，初始随机放置 2 个数字（2 的概率 0.9，4 的概率 0.1）；
- 每次滑动后（棋盘发生变化才算有效步）随机生成一个新数字；
- 相同数字相撞合并，每对每步只合并一次；
- 无任何合法移动时游戏结束。

动作空间：Discrete(4)  0=左, 1=右, 2=上, 3=下
观测空间：Box(0, 1, (144,))  每个格子做 log2 值的 one-hot 编码（16 个 bin），9 * 16 = 144 维
奖励：本步合并产生的得分（合出新格子的数值之和）；无效动作奖励 -1（训练时使用动作掩码，不会触发）。

本环境动作语义与课程提供的 game2048 包保持一致（LEFT/RIGHT/UP/DOWN）。
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_MEANING = {0: "left", 1: "right", 2: "up", 3: "down"}
NUM_CHANNELS = 16  # one-hot 的 bin 数：空格=0，格子 2^k 占第 k 个 bin（k 最大 15）
MAX_LOG2 = NUM_CHANNELS - 1


def _slide_row_left(row):
    """把一行向左滑动，返回 (新行, 本行合并得分)。"""
    nonzero = [x for x in row if x != 0]
    out, gain = [], 0
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
            v = nonzero[i] * 2
            out.append(v)
            gain += v
            i += 2
        else:
            out.append(nonzero[i])
            i += 1
    out.extend([0] * (len(row) - len(out)))
    return out, gain


def _can_slide_row_left(row):
    """该行能否向左滑动：非零数左边有空格，或相邻两个非零数相等。"""
    gap = False
    prev = 0
    for x in row:
        if x == 0:
            gap = True
        else:
            if gap or x == prev:
                return True
            prev = x
    return False


class Game2048Env3x3(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 3):
        super().__init__()
        assert size == 3, "本实现针对 3x3 棋盘优化（纯 python 滑动逻辑）"
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(size * size * NUM_CHANNELS,), dtype=np.float32
        )
        self.board = [[0] * size for _ in range(size)]
        self.score = 0  # 累计合并得分

    # ------------------------------------------------------------------ API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self._spawn()
        self._spawn()
        return self._encode(), self._info()

    def step(self, action: int):
        moved, gain = self._apply(int(action))
        if moved:
            self._spawn()
            self.score += gain
            reward = float(gain)
            invalid = False
        else:
            reward = -1.0
            invalid = True
        terminated = not any(self.action_mask())
        truncated = False
        return self._encode(), reward, terminated, truncated, self._info(invalid)

    def action_mask(self):
        """长度为 4 的 bool 列表：每个方向是否为合法（能改变棋盘的）动作。"""
        cols = [[self.board[r][c] for r in range(self.size)] for c in range(self.size)]
        mask = [False] * 4
        for row in self.board:
            mask[0] = mask[0] or _can_slide_row_left(row)
            mask[1] = mask[1] or _can_slide_row_left(row[::-1])
        for col in cols:
            mask[2] = mask[2] or _can_slide_row_left(col)
            mask[3] = mask[3] or _can_slide_row_left(col[::-1])
        return mask

    def render(self):
        print("\n".join(" ".join(f"{x:5d}" for x in row) for row in self.board))

    # ------------------------------------------------------------- internals
    def _apply(self, action: int):
        """执行滑动，返回 (棋盘是否变化, 合并得分)。"""
        gain = 0
        moved = False
        if action == 0:  # left
            for i in range(self.size):
                new, g = _slide_row_left(self.board[i])
                gain += g
                moved = moved or new != self.board[i]
                self.board[i] = new
        elif action == 1:  # right
            for i in range(self.size):
                new, g = _slide_row_left(self.board[i][::-1])
                gain += g
                new = new[::-1]
                moved = moved or new != self.board[i]
                self.board[i] = new
        elif action == 2:  # up
            for c in range(self.size):
                col = [self.board[r][c] for r in range(self.size)]
                new, g = _slide_row_left(col)
                gain += g
                moved = moved or new != col
                for r in range(self.size):
                    self.board[r][c] = new[r]
        elif action == 3:  # down
            for c in range(self.size):
                col = [self.board[r][c] for r in range(self.size)]
                new, g = _slide_row_left(col[::-1])
                gain += g
                new = new[::-1]
                moved = moved or new != col
                for r in range(self.size):
                    self.board[r][c] = new[r]
        else:
            raise ValueError(f"非法动作: {action}")
        return moved, gain

    def _spawn(self):
        empty = [(r, c) for r in range(self.size) for c in range(self.size)
                 if self.board[r][c] == 0]
        if not empty:
            return
        r, c = empty[self.np_random.integers(len(empty))]
        self.board[r][c] = 2 if self.np_random.random() < 0.9 else 4

    def _encode(self) -> np.ndarray:
        """(3,3) 棋盘 -> 144 维 one-hot 特征。"""
        feats = np.zeros(self.size * self.size * NUM_CHANNELS, dtype=np.float32)
        for r in range(self.size):
            for c in range(self.size):
                x = self.board[r][c]
                k = min(x.bit_length() - 1, MAX_LOG2) if x else 0
                feats[(r * self.size + c) * NUM_CHANNELS + k] = 1.0
        return feats

    def _info(self, invalid: bool = False) -> dict:
        flat = [x for row in self.board for x in row]
        return {
            "invalid": invalid,
            "board": [row[:] for row in self.board],
            "max_tile": max(flat),
            "board_sum": sum(flat),
            "score": self.score,
        }


if __name__ == "__main__":
    # 简单自检
    env = Game2048Env3x3()
    obs, info = env.reset(seed=0)
    assert obs.shape == (144,) and obs.dtype == np.float32
    done, steps = False, 0
    while not done:
        mask = env.action_mask()
        assert any(mask)
        obs, r, done, trunc, info = env.step(int(np.flatnonzero(mask)[0]))
        steps += 1
        assert info["max_tile"] >= 2
    print(f"自检通过：随机合法动作共存活 {steps} 步, "
          f"最大块 {info['max_tile']}, 棋盘总和 {info['board_sum']}, 得分 {info['score']}")
    # 合并逻辑样例
    assert _slide_row_left([2, 2, 0]) == ([4, 0, 0], 4)
    assert _slide_row_left([2, 2, 2]) == ([4, 2, 0], 4)
    assert _slide_row_left([4, 2, 2]) == ([4, 4, 0], 4)
    assert _slide_row_left([2, 0, 2]) == ([4, 0, 0], 4)
    assert _slide_row_left([4, 2, 0]) == ([4, 2, 0], 0)
    print("合并逻辑断言全部通过")
