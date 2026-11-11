"""1 万局大样本终测：种子 70000-79999，best.pt 贪心，GPU 批量。"""
import csv, json, os, time
import numpy as np
import torch
from ppo import ActorCritic
from game2048_env_fast import FastGame2048Batch

def wilson(p, n, z=1.96):
    den = 1 + z*z/n
    ctr = p + z*z/(2*n)
    hw = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (ctr - hw)/den, (ctr + hw)/den

def eval_ckpt(ckpt_path, tag, out_dir, n=10000, seed=70000, device="cuda", batch=1000):
    model = ActorCritic(hidden=256).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()
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
    ln_arr = np.array([r["length"] for r in rows])
    res = {"tag": tag, "ckpt": ckpt_path, "games": n,
           "board_sum_mean": float(bs_arr.mean()), "board_sum_std": float(bs_arr.std()),
           "score_mean": float(np.array([r["score"] for r in rows]).mean()),
           "length_mean": float(ln_arr.mean()),
           "p>=128": float((mt_arr >= 128).mean()),
           "p>=256": float((mt_arr >= 256).mean()),
           "p>=512": float((mt_arr >= 512).mean()),
           "p>=1024": float((mt_arr >= 1024).mean()),
           "p>=2048": float((mt_arr >= 2048).mean()),
           "seconds": round(time.time() - t0, 1)}
    for key in ("p>=256", "p>=512", "p>=1024"):
        lo, hi = wilson(res[key], n)
        res[key + "_ci95"] = [round(lo, 4), round(hi, 4)]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, tag + "_10k_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "score", "board_sum", "max_tile", "length"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, tag + "_10k_summary.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("%-24s sum=%6.1f p256=%.4f [%.4f,%.4f] p512=%.4f p1024=%.4f (%.0fs)" % (
        tag, res["board_sum_mean"], res["p>=256"], *res["p>=256_ci95"],
        res["p>=512"], res["p>=1024"], res["seconds"]), flush=True)

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    RUNS = [
        ("ref2x_ent2_s2", "results/budget_extension/epochs10_lr625e4_s2/best.pt"),
        ("ref2x_ent2_s3", "results/budget_extension/epochs10_lr625e4_s3/best.pt"),
        ("ref2x_ent2_s4", "results/budget_extension/epochs10_lr625e4_s4/best.pt"),
        ("gamma_g998_s2", "results/gamma_g998_s2/best.pt"),
        ("gamma_g998_s3", "results/gamma_g998_s3/best.pt"),
        ("gamma_g998_s4", "results/gamma_g998_s4/best.pt"),
        ("ft_ent8_s3", "results/budget4x_ft_ent8_s3/best.pt"),
    ]
    os.chdir(base)
    for tag, ckpt in RUNS:
        eval_ckpt(ckpt, tag, "results/test10k")
    print("10K_DONE")
