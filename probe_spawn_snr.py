"""探针：拥挤局面下 动作对比信号 vs spawn 运气噪声 的定量测量。

对每个采样到的拥挤状态 s、每个合法动作 a：
  - 采 K 条续局（每条从 a 出发，后续贪心，spawn 按真实分布随机）
  - 每条算 n 步折扣回报 + bootstrap V（H 步）
信号 signal(a) = Q̂(s,a) - mean_a Q̂   （策略需要分辨的东西）
噪声 noise(a) = std_K(Q̂ 的 K 次采样)  （spawn 运气 + 采样噪声）
"""
import csv, os, time
import numpy as np
import torch
from ppo import ActorCritic
from game2048_env_fast import FastGame2048Batch

CKPT = "results/budget4x_ft_ent8_s3/best.pt"
H = 32
K = 8
GAMMA = 0.995

model = ActorCritic(hidden=256).to("cuda")
sd = torch.load(CKPT, map_location="cuda", weights_only=False)
model.load_state_dict(sd["model"])
model.eval()

# 1) 自对弈收集拥挤状态（去重按棋盘等级串）
seen = {}
env = FastGame2048Batch(512, seed=910)
rng = np.random.default_rng(910)
for t in range(600):
    mask = torch.from_numpy(env.mask).to("cuda")
    obs = torch.from_numpy(env.obs).to("cuda")
    act, _, _ = model.act_and_value(obs, mask, deterministic=True)
    cells = env.cells.copy()
    for i in range(512):
        b = tuple(cells[i].tolist())
        lv = int(cells[i].max())
        occ = int((cells[i] > 0).sum())
        if lv >= 8 and occ >= 6 and b not in seen:
            seen[b] = (cells[i].copy(), int(env.mask[i].sum()))
    env.step(act.cpu().numpy())
    if len(seen) >= 400:
        break
states = list(seen.items())
print(f"收集到去重拥挤状态 {len(states)} 个")

# 2) 分支测量：每状态 x 每合法首动作 x K 条续局
def batched_eval(boards, per_board_actions):
    """boards: list of (9,) levels；per_board_actions: list of list[int]。
    返回 per_board: list of (n_actions, K) 折扣回报矩阵。"""
    flat_boards, flat_actions = [], []
    spans = []
    for bi, (b, acts) in enumerate(zip(boards, per_board_actions)):
        spans.append((len(flat_boards), len(acts)))
        for a in acts:
            for _ in range(K):
                flat_boards.append(b)
                flat_actions.append(a)
    N = len(flat_boards)
    env = FastGame2048Batch(N)
    env.set_states(np.arange(N), np.array(flat_boards, dtype=np.int64))
    obs = torch.from_numpy(env.obs).to("cuda")
    mask = torch.from_numpy(env.mask).to("cuda")
    R = np.zeros(N)
    disc = np.ones(N)
    alive = np.ones(N, bool)
    # 第一步用指定动作
    env.step(np.array(flat_actions, dtype=np.int64))
    for h in range(H - 1):
        obs = torch.from_numpy(env.obs).to("cuda")
        mask = torch.from_numpy(env.mask).to("cuda")
        act, _, v = model.act_and_value(obs, mask, deterministic=True)
        rew, done = env.step(act.cpu().numpy())
        R += disc * alive * rew
        disc *= GAMMA
        alive &= (done == 0)
    obs = torch.from_numpy(env.obs).to("cuda")
    mask = torch.from_numpy(env.mask).to("cuda")
    _, _, v_final = model.act_and_value(obs, mask, deterministic=True)
    R += disc * alive * v_final.cpu().numpy()
    out = []
    for bi, (start, na) in enumerate(spans):
        Q = R[start:start + na * K].reshape(na, K)
        out.append(Q)
    return out

t0 = time.time()
rows = []
n_probe = 0
for bi, (bkey, (cells, n_legal)) in enumerate(states[:48]):
    b = tuple(int(x) for x in cells)
    env1 = FastGame2048Batch(1)
    env1.set_states(np.array([0]), cells[None, :])
    legal = np.flatnonzero(env1.mask[0])
    if len(legal) < 2:
        continue
    boards, acts = [], []
    for a in legal:
        boards.append(b)
        acts.append(int(a))
    Qs = batched_eval([b], [acts])[0]
    mean_a = Qs.mean(axis=1)
    noise_a = Qs.std(axis=1)
    signal = mean_a.max() - mean_a.mean()
    noise = noise_a.mean()
    occ = int((cells > 0).sum())
    rows.append((signal, noise, occ, n_legal))
    n_probe += 1
    if n_probe % 8 == 0:
        print(f"  已探 {n_probe} 态 ({time.time()-t0:.0f}s)")

arr = np.array(rows)
print("探针状态数: {len(arr)}  (每态 4 动作 x {K} spawn 采样 x {H} 步)")
print(f"动作对比信号 mean_a 的极差: {arr[:,0].mean():.1f} ± {arr[:,0].std():.1f}")
print(f"spawn 运气噪声 (组内std):   {arr[:,1].mean():.1f} ± {arr[:,1].std():.1f}")
print(f"信噪比 SNR = 信号/噪声: {arr[:,0].mean()/arr[:,1].mean():.3f}")
for occ in (6, 7, 8):
    m = arr[arr[:,2] == occ]
    if len(m):
        print(f"  占用={occ}: 信号 {m[:,0].mean():.1f} | 噪声 {m[:,1].mean():.1f} | SNR {m[:,0].mean()/m[:,1].mean():.3f} (n={len(m)})")
