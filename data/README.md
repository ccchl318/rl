# 期中展示数据包

对应课件：`pptx/强化学习期中展示.pptx`（13 页）

本目录只收录**课件结论直接依赖**的数据与训练参数。
无关的消融实验（clip 扫描、熵扫描、γ 扫描、分支采样、容量对比、奖励塑形等）未收录。

---

## 一、课件每一页 → 对应文件

| 页 | 内容 | 数据来源 |
|---:|---|---|
| 6 | 三阶段学习率调度 | `logs/ultrafine_s*/ultrafine_s*_eval_log.csv`（含 lr 列）|
| 7 | EMA 原理 | 无需数据（概念页）|
| 8 | score 与棋盘和随 iter 变化 | `logs/validation_s*/eval_log.csv`（每 250 轮 500 局）|
| 9 | 三个阈值的达成率曲线 | 同上（`p>=128/256/512` 列）|
| 10 | 学习率调度对照 | `logs/ultrafine_s*.csv` `ext2x_s*.csv` `budget4x_s*.csv` `fixedlr_s*.csv` |
| 11 | 最终结果（753.5 / 5050 / 97.6% / 67.2% / 0.37%）| `eval/ultrafine_s*_test_episodes.csv` + `summary.json` |
| 12 | 两局真实对局 | `videos/seed_70151.mp4`、`videos/seed_70876.mp4`（模型 `checkpoints/combo_s2.pt`）|
| — | 大样本选模复核（10,000 局）| `eval/validation_10k/*.csv` + `validation_10k_report.md` |

> 所有均值/达标率都可由 `eval/` 下的**逐局 CSV**（seed / score / board_sum / max_tile / length）直接复算。

## 二、检查点

格式统一为 `torch.save({"args", "it", "model", "optimizer"})`，`hidden=256`：

~~~python
import torch
from ppo import ActorCritic
ck = torch.load("checkpoints/ultrafine_s2.pt", map_location="cpu", weights_only=False)
model = ActorCritic(hidden=ck["args"]["hidden"])
model.load_state_dict(ck["model"]); model.eval()
print("训练到第", ck["it"], "次迭代")
~~~

| 文件 | 说明 |
|---|---|
| `ultrafine_s2/s3/s4.pt` | **本方案最终模型**：三阶段调度 + EMA，18,000 迭代，三个训练种子 |
| `combo_s2.pt` | 课件第 12 页两段录像所用的模型 |

三个种子的 1,000 局终测（种子 70000–70999）：

| 种子 | 棋盘和 | P(≥256) | P(≥512) | P(≥1024) |
|---|---:|---:|---:|---:|
| s2 | 768.6 | 97.8% | 69.2% | 0.90% |
| s3 | 753.0 | 97.6% | 67.5% | 0.10% |
| s4 | 738.8 | 97.3% | 65.0% | 0.10% |
| **均值** | **753.5** | **97.6%** | **67.2%** | **0.37%** |

## 三、复现评测

~~~bash
python evaluate.py --ckpt data/checkpoints/ultrafine_s2.pt --episodes 1000 --seed 70000 --env fast
~~~

口径：贪心（argmax）、固定种子、3×3 棋盘、新块 2:4 = 0.9:0.1。

## 四、目录结构

~~~
data/
├── pptx/          课件原件（13 页）
├── checkpoints/   4 个网络参数 .pt
├── logs/          训练/验证曲线 CSV（含 lr、熵、KL 列）
├── eval/          逐局终测数据 + 10,000 局复核
├── videos/        两局对局录像
└── README.md
~~~
