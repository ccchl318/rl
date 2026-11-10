"""gamma 矩阵终测（本地版）：每模型 1000 局，种子 70000 起，best.pt 贪心。"""
import csv, json, os, time
import numpy as np
import torch
from ppo import ActorCritic
from game2048_env_fast import FastGame2048Batch

def eval_ckpt(ckpt_path, tag, out_dir, n=1000, seed=70000, device="cuda", batch=500):
    model = ActorCritic(hidden=256).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()
    best_it = sd.get("it")
    rows = []
    t0 = time.time()
    n_done = 0
    while n_done < n:
        b = min(batch, n - n_done)
        env = FastGame2048Batch(b, seeds=[seed + n_done + i for i in range(b)])
        alive = np.ones(b, dtype=bool)
        while alive.any():
            obs = torch.from_numpy(env.obs).to(device)
            mask = torch.from_numpy(env.mask).to(device)
            act, _, _ = model.act_and_value(obs, mask, deterministic=True)
            env.step(act.cpu().numpy())
            for i, stat in env.pop_finished():
                if alive[i]:
                    alive[i] = False
                    sc, mt, bs, ln = stat
                    rows.append({"seed": seed + n_done + i, "score": sc,
                                 "board_sum": bs, "max_tile": mt, "length": ln})
        n_done += b
    mt_arr = np.array([r["max_tile"] for r in rows])
    bs_arr = np.array([r["board_sum"] for r in rows])
    summary = {
        "tag": tag, "ckpt": ckpt_path, "best_iter": best_it,
        "board_sum_mean": float(bs_arr.mean()), "board_sum_std": float(bs_arr.std()),
        "p>=256": float((mt_arr >= 256).mean()), "p>=512": float((mt_arr >= 512).mean()),
        "p>=1024": float((mt_arr >= 1024).mean()), "p>=2048": float((mt_arr >= 2048).mean()),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, tag + "_test_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "score", "board_sum", "max_tile", "length"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, tag + "_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("%-24s sum=%6.1f p256=%.3f p512=%.3f p1024=%.4f best_it=%s" % (
        tag, summary["board_sum_mean"], summary["p>=256"], summary["p>=512"],
        summary["p>=1024"], best_it), flush=True)

if __name__ == "__main__":
    RUNS = [(f"gamma_g997_s{s}", f"results/gamma_g997_s{s}/best.pt") for s in (2, 3, 4)] \
         + [(f"gamma_g998_s{s}", f"results/gamma_g998_s{s}/best.pt") for s in (2, 3, 4)] \
         + [(f"gamma_g998_t256_s{s}", f"results/gamma_g998_t256_s{s}/best.pt") for s in (2, 3, 4)]
    for tag, ckpt in RUNS:
        eval_ckpt(ckpt, tag, "results/gamma_eval")
