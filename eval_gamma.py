"""gamma 矩阵终测：每模型 1000 局，种子 70000 起，best.pt 贪心。"""
from eval4x import eval_ckpt

RUNS = []
for s in (2, 3, 4):
    RUNS.append((f"gamma_g997_s{s}", f"results/gamma_g997_s{s}/best.pt"))
for s in (2, 3, 4):
    RUNS.append((f"gamma_g998_s{s}", f"results/gamma_g998_s{s}/best.pt"))
for s in (2, 3, 4):
    RUNS.append((f"gamma_g998_t256_s{s}", f"results/gamma_g998_t256_s{s}/best.pt"))

if __name__ == "__main__":
    for tag, ckpt in RUNS:
        eval_ckpt(ckpt, tag, "results/gamma_eval")
