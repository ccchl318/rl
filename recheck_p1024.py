"""核查 P(>=1024) 的估计：同一检查点换不同测试种子各跑 2000 局。"""
import numpy as np
from ppo import ActorCritic
from game2048_env_fast import FastGame2048Batch
import torch
import sys

ckpt = sys.argv[1] if len(sys.argv) > 1 else "results/ultrafine_s2/best.pt"
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
model = ActorCritic(hidden=sd["args"]["hidden"])
model.load_state_dict(sd["model"])
model.eval()
print("checkpoint:", ckpt, "| it =", sd["it"])

def run(seed_base, n, batch=500):
    tiles = []
    n_done = 0
    while n_done < n:
        b = min(batch, n - n_done)
        env = FastGame2048Batch(b, seeds=[seed_base + n_done + i for i in range(b)])
        alive = np.ones(b, dtype=bool)
        while alive.any():
            obs = torch.from_numpy(env.obs)
            mask = torch.from_numpy(env.mask)
            act, _, _ = model.act_and_value(obs, mask, deterministic=True)
            env.step(act.numpy())
            for i, stat in env.pop_finished():
                if alive[i]:
                    alive[i] = False
                    tiles.append(stat[1])
        n_done += b
    return np.array(tiles)

for base in (70000, 3000000, 5000000):
    t = run(base, 2000)
    print("  seeds %8d..: 棋盘和 %6.1f | P256 %.3f | P512 %.3f | P1024 %d 局 (%.3f%%)" % (
        base, t.mean(), (t >= 256).mean(), (t >= 512).mean(),
        (t >= 1024).sum(), (t >= 1024).mean() * 100))
