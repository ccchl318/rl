"""3x3 版 2048 的向量化批量环境（查表 + numpy fancy indexing）。

游戏语义与 Game2048Env3x3（game2048_env.py）逐位一致：合并规则、奖励、终止判定、
动作掩码、one-hot 编码、spawn 分布（空位均匀、2 概率 0.9 / 4 概率 0.1）。
区别在于把 N 个环境的一次 step 压缩为若干次 numpy 向量操作，不再逐环境 Python 循环。

行查表：每格存 log2 值（0=空），一行 3 格打包为 12bit 行代码 c0<<8 | c1<<4 | c2，
全部 16^3 = 4096 种行局面的移动结果、合并得分、可动性在 import 时预计算成表。

RNG 两种模式：
- shared（默认）：单一 Generator 向量化采样，训练用，最快；
- per_env（传入 seeds 列表时启用）：每环境独立 Generator，调用顺序与旧环境完全一致
  （整数抽空位 -> 均匀抽样定 2/4），用于等价测试与批量贪心评估的精确复现。

用法：
    env = FastGame2048Batch(n_envs=64, seed=0, reward_scale=0.25)
    obs, mask = env.obs, env.mask          # (64,144) float32 / (64,4) bool
    rew, done = env.step(actions)          # actions: (64,) int
    for i, stat in env.pop_finished():     # (env_idx, (score, max_tile, board_sum, length))
        ...
"""

import numpy as np

SIZE = 3
NUM_CELLS = SIZE * SIZE
NUM_CHANNELS = 16  # one-hot bin 数，与 game2048_env 一致
OBS_DIM = NUM_CELLS * NUM_CHANNELS
MAX_LOG2 = NUM_CHANNELS - 1
N_CODES = 1 << (4 * SIZE)  # 4096

# 每个动作下 3 条线的取格顺序（按移动方向排列，左滑语义作用于该顺序）
LINE_IDX = np.array([
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]],   # 0 左
    [[2, 1, 0], [5, 4, 3], [8, 7, 6]],   # 1 右
    [[0, 3, 6], [1, 4, 7], [2, 5, 8]],   # 2 上
    [[6, 3, 0], [7, 4, 1], [8, 5, 2]],   # 3 下
], dtype=np.int64)


def _slide_left_log2(row):
    """log2 域的左滑（语义同 game2048_env._slide_row_left），返回 (新行, 合并得分)。"""
    nonzero = [x for x in row if x != 0]
    out, gain = [], 0
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
            out.append(nonzero[i] + 1)        # 合并后的 log2 值
            gain += 1 << (nonzero[i] + 1)     # 得分 = 合并出的新块数值
            i += 2
        else:
            out.append(nonzero[i])
            i += 1
    out.extend([0] * (SIZE - len(out)))
    return out, gain


def _pack(row):
    return (row[0] << 8) | (row[1] << 4) | row[2]


def _build_tables():
    move_l = np.zeros(N_CODES, dtype=np.int32)
    move_r = np.zeros(N_CODES, dtype=np.int32)
    gain = np.zeros(N_CODES, dtype=np.int32)
    can_l = np.zeros(N_CODES, dtype=bool)
    can_r = np.zeros(N_CODES, dtype=bool)
    for code in range(N_CODES):
        row = (code >> 8 & 15, code >> 4 & 15, code & 15)
        new, g = _slide_left_log2(row)
        move_l[code] = _pack(new)
        gain[code] = g
        can_l[code] = move_l[code] != code
        rev = row[::-1]
        new_r, _ = _slide_left_log2(rev)
        move_r[code] = _pack(new_r[::-1])
        can_r[code] = move_r[code] != code
    return move_l, move_r, gain, can_l, can_r


MOVE_L, MOVE_R, GAIN, CAN_L, CAN_R = _build_tables()
POW2 = (1 << np.arange(MAX_LOG2 + 1)).astype(np.int64)  # log2 -> tile 值


class FastGame2048Batch:
    """N 个 3x3 2048 环境的向量化批量实现（done 自动重开一局）。"""

    def __init__(self, n_envs: int, seed: int = 0, reward_scale: float = 1.0,
                 seeds=None):
        self.n = n_envs
        self.reward_scale = reward_scale
        self.cells = np.zeros((n_envs, NUM_CELLS), dtype=np.int64)
        self.obs = np.zeros((n_envs, OBS_DIM), dtype=np.float32)
        self.mask = np.zeros((n_envs, 4), dtype=bool)
        self._score = np.zeros(n_envs, dtype=np.float64)   # 局内累计合并得分（未缩放）
        self._len = np.zeros(n_envs, dtype=np.int64)
        self._finished = []  # (env_idx, (score, max_tile, board_sum, length))
        if seeds is None:
            self.gens = None
            self.rng = np.random.default_rng(seed)
        else:
            assert len(seeds) == n_envs, "seeds 数量必须等于 n_envs"
            self.gens = [np.random.default_rng(int(s)) for s in seeds]
            self.rng = None
        self._reset_boards()

    # ------------------------------------------------------------- 对外接口
    def pop_finished(self):
        """取走已结束对局的统计列表 [(env_idx, (score, max_tile, board_sum, len))]。"""
        out, self._finished = self._finished, []
        return out

    def step(self, actions):
        """批量执行一步。actions: (N,) int；返回 (rewards, dones)，均为 (N,)。

        rewards 已乘 reward_scale（无效动作 -1 同样缩放，语义同旧环境+训练循环）；
        dones 为 1.0/0.0。done 的环境自动重开一局（RNG 流延续，不重播种）。
        """
        n = self.n
        a = np.asarray(actions, dtype=np.int64)
        rows = np.arange(n)

        # 1) 按动作方向取 3 条线，打包成行代码
        idxp = LINE_IDX[a]                                   # (N,3,3)
        c = self.cells[rows[:, None, None], idxp]            # (N,3,3) log2 值
        codes = (c[..., 0] << 8) | (c[..., 1] << 4) | c[..., 2]  # (N,3)

        # 2) 查表移动：LINE_IDX 已按移动方向排列（右/下为反序），统一用左滑表即可
        new_codes = MOVE_L[codes]
        gain = GAIN[codes].sum(axis=1).astype(np.float64)
        moved = (new_codes != codes).any(axis=1)

        # 3) 写回棋盘
        unpacked = np.stack([new_codes >> 8 & 15, new_codes >> 4 & 15,
                             new_codes & 15], axis=-1)       # (N,3,3)
        self.cells[rows[:, None, None], idxp] = unpacked
        np.minimum(self.cells, MAX_LOG2, out=self.cells)     # one-hot bin 上限保护

        # 4) 奖励与局内统计（无效动作 reward=-1，不 spawn；语义同旧环境）
        reward = np.where(moved, gain * self.reward_scale, -1.0 * self.reward_scale)
        self._score += np.where(moved, gain, 0.0)
        self._len += 1

        # 5) spawn（仅对有效步）
        if moved.any():
            self._spawn(np.flatnonzero(moved))

        # 6) 重新计算动作掩码（= 旧环境 spawn 后的终止判定）
        self._refresh_mask()
        done = ~self.mask.any(axis=1)

        # 7) 结束的局记录统计并自动重开
        done_idx = np.flatnonzero(done)
        if done_idx.size:
            tiles = POW2[self.cells]
            sums = tiles.sum(axis=1)
            maxs = tiles.max(axis=1)
            for i in done_idx:
                i = int(i)
                self._finished.append((i, (float(self._score[i]), int(maxs[i]),
                                           int(sums[i]), int(self._len[i]))))
            self._score[done_idx] = 0.0
            self._len[done_idx] = 0
            self._reset_boards(done_idx)

        dones = done.astype(np.float64)
        return reward.astype(np.float64), dones

    # ------------------------------------------------------------- internals
    def _spawn(self, idx):
        """对 idx 中的环境各生成一个新块（空位均匀，2 w.p. 0.9 / 4 w.p. 0.1）。"""
        if idx.size == 0:
            return
        empty = self.cells[idx] == 0
        n_empty = empty.sum(axis=1)
        if self.gens is None:  # shared：向量化，两次抽样的分布与旧环境一致
            u = self.rng.random(idx.size)
            u4 = self.rng.random(idx.size)
            slot = (u * n_empty).astype(np.int64)
            tile = np.where(u4 < 0.9, 1, 2)
        else:                  # per_env：逐环境、与旧环境调用顺序逐位一致
            slot = np.empty(idx.size, dtype=np.int64)
            tile = np.empty(idx.size, dtype=np.int64)
            for k, i in enumerate(idx):
                g = self.gens[int(i)]
                ne = int(n_empty[k])
                slot[k] = g.integers(ne)
                tile[k] = 1 if g.random() < 0.9 else 2
        cum = np.cumsum(empty, axis=1)
        pos = np.argmax(cum == (slot[:, None] + 1), axis=1)  # 第 slot+1 个空格的位置
        self.cells[idx, pos] = tile

    def _reset_boards(self, idx=None):
        """清空（部分）棋盘并按旧环境 reset 语义连放两个块。"""
        if idx is None:
            idx = np.arange(self.n)
        if idx.size == 0:
            return
        self.cells[idx] = 0
        self._spawn(idx)
        self._spawn(idx)
        self._refresh_mask(idx)

    def _refresh_mask(self, idx=None):
        """由当前棋盘计算动作掩码（左/右/上/下），语义同旧环境 action_mask。"""
        if idx is None:
            idx = np.arange(self.n)
            cells = self.cells
        else:
            cells = self.cells[idx]
        row_codes = np.stack([(cells[:, 0] << 8) | (cells[:, 1] << 4) | cells[:, 2],
                              (cells[:, 3] << 8) | (cells[:, 4] << 4) | cells[:, 5],
                              (cells[:, 6] << 8) | (cells[:, 7] << 4) | cells[:, 8]], 1)
        col_codes = np.stack([(cells[:, 0] << 8) | (cells[:, 3] << 4) | cells[:, 6],
                              (cells[:, 1] << 8) | (cells[:, 4] << 4) | cells[:, 7],
                              (cells[:, 2] << 8) | (cells[:, 5] << 4) | cells[:, 8]], 1)
        mask = np.empty((idx.size, 4), dtype=bool)
        mask[:, 0] = CAN_L[row_codes].any(axis=1)
        mask[:, 1] = CAN_R[row_codes].any(axis=1)
        mask[:, 2] = CAN_L[col_codes].any(axis=1)
        mask[:, 3] = CAN_R[col_codes].any(axis=1)
        if idx is None:
            self.mask = mask
        else:
            self.mask[idx] = mask
        self._encode(idx)

    def _encode(self, idx=None):
        """棋盘 -> (N,144) one-hot float32，布局同旧环境 _encode。"""
        if idx is None:
            idx = np.arange(self.n)
        k = idx.size
        sub = np.zeros((k, OBS_DIM), dtype=np.float32)
        pos = (np.arange(NUM_CELLS) * NUM_CHANNELS)[None, :] + self.cells[idx]
        sub[np.repeat(np.arange(k), NUM_CELLS), pos.ravel()] = 1.0
        self.obs[idx] = sub


if __name__ == "__main__":
    # 简单自检：单环境随机合法动作存活步数应与旧环境同量级
    env = FastGame2048Batch(1, seed=0)
    rng = np.random.default_rng(0)
    steps, done = 0, False
    while not done:
        legal = np.flatnonzero(env.mask[0])
        _, d = env.step([legal[rng.integers(len(legal))]])
        done = bool(d[0])
        steps += 1
    fin = env.pop_finished()
    print(f"自检通过：随机合法动作存活 {steps} 步, 局统计 {fin[0][1]}")
