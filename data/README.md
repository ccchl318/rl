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

---

## 五、课件配图的原始文件（data/figures/）

课件里每一张图都单独存了一份 PNG，可直接引用或重新排版：

| 文件 | 对应页 | 内容 |
|---|---|---|
| s02_method_taxonomy.png | 第 2 页 | 强化学习分类树（PPO vs n-tuple TD）|
| s03_task_model.png | 第 3 页 | 3×3 任务建模 |
| s04_dual_tower.png | 第 4 页 | 双塔网络结构图 |
| s04_clip_formula.png | 第 4 页 | PPO-clip 目标函数公式 |
| s04_clip_chart.png | 第 4 页 | clip 的直观示意（Â>0 / Â<0）|
| s05_hyperparams.png | 第 5 页 | 超参数设置表 |
| s06_lr_schedule.png | 第 6 页 | 三阶段学习率调度曲线 |
| s07_ema.png | 第 7 页 | EMA 原理示意 |
| s08_training_curves.png | 第 8 页 | score 与棋盘和随迭代变化 |
| s09_thresholds.png | 第 9 页 | 128 / 256 / 512 三档达成率 |
| s10_lr_comparison.png | 第 10 页 | 四种学习率调度对照 |
| s12_game70151_poster.png | 第 12 页 | 典型局（第 200 步，录像封面）|
| s12_game70151_final.png | 第 12 页 | 典型局终局棋盘 |
| s12_game70876_poster.png | 第 12 页 | 1024 局（第 400 步，录像封面）|
| s12_game70876_final.png | 第 12 页 | 1024 局终局棋盘 |

**这些图都是脚本生成的**，可复现：

| 生成脚本 | 产出 |
|---|---|
| make_best_figs.py | figures/best/ 下 6 张报告图（fig1–fig6）|
| render_videos.py | videos/ 下的对局 MP4；单帧可导出为 PNG |
| eval_hidden.py | 任意检查点的 1,000 局终测（宽度自适应），产出 eval/ 下 CSV |

> 说明：s02–s10 这些图是用排版脚本绘制的（文字/坐标轴为矢量转位图），
> 其**数据来源**见下表"数据来源"列；图形本身可用 data/figures/ 中的 PNG 直接复用。

---

## 六、结果数据的完整查询路径

| 想查什么 | 去哪看 |
|---|---|
| 某个模型的逐局结果 | eval/*_test_episodes.csv（列：seed, score, board_sum, max_tile, length）|
| 各模型汇总指标 | eval/*_summary.json |
| 训练过程曲线（含 lr / 熵 / KL）| logs/**/eval_log.csv、logs/**/train_log.csv |
| 10,000 局大样本复核 | eval/validation_10k/*.csv（每模型一个文件）|
| 课件结论与数据的对应关系 | 本文件第一节的表格 |
| 完整实验报告 | 仓库根目录 BEST_CONFIG.md（当前最优）|
