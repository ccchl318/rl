# PPO 实验整理包

本目录是 3×3 2048 PPO 实验的**自包含**整理包：代码、文档、图表与全部实验结果
集中在此，可独立浏览、训练与复现，不依赖上级目录。

技术细节见 [PPO_IMPLEMENTATION.md](PPO_IMPLEMENTATION.md)（状态/奖励/网络/训练
超参数/数据表，全部与代码逐项核对）。

## 文件索引

| 路径 | 内容 |
|---|---|
| `game2048_env.py` | 3×3 环境（Gymnasium 接口：状态编码、奖励、动作掩码、滑动合并） |
| `ppo.py` | PPO 核心：ActorCritic 网络、GAE、PPO-clip 更新（含价值裁剪）、贪心评估 |
| `train.py` | 训练脚本：同步向量化采集 → GAE → PPO 更新 → 周期验证选模 |
| `evaluate.py` | 独立评估：加载 checkpoint 跑贪心对局，输出统计、分布图、最优局棋谱 |
| `run_sensitivity_experiments.py` | 超参数敏感性实验：15 配置（14 单变量 + 1 合并配置）× 3 种子批量训练（单变量 OFAT，自动检测 GPU / `--devices cpu` 回退） |
| `compare_sensitivity.py` | 敏感性评测：42 个 best.pt 各 1000 局独立测试，生成对比表、排序图、曲线图与报告 |
| `make_ppo_report_figs.py` | 从 `results/` 现有数据重新生成 `figures/` 全部 7 张图 |
| `PPO_IMPLEMENTATION.md` | 完整实现文档 |
| `figures/fig1–fig7*.png` | 训练流程、状态编码、网络结构、奖励定义、验证曲线、训练统计、测试分布 |
| `results/main/` | **主实验**（24,576,000 步，种子 1）：`best.pt`/`last.pt` 检查点、`train_log.csv`、`eval_log.csv`、评估报告与最优局录像 |
| `results/ablation_*/` | **消融实验**（短预算 328 万步 × 3 种子）：default（基准）、g099（γ=0.99）、lowent（低熵系数）、nomask（去动作掩码），另含 `ablation_summary.{txt,csv,png}` 汇总与各次训练的控制台日志（`.out`） |
| `results/sensitivity/` | **超参数敏感性实验**（完整预算 24,576,000 步 × 15 配置 = 47 次训练；最优配置 5 个种子，其余各 3 个）：`report.md` 主表与解读、`comparison.json`、5 万局逐局数据、`sensitivity_bars.png` 排序图、`sensitivity_curves.png` 学习曲线；47 个运行目录保留 `best.pt` 与训练日志（`last.pt` 未随包以控制体积） |
| `results/budget_extension/` | **预算充分性检验**：最优配置 × 49,152,000 步（2 倍预算）× 3 种子，含 `report.md` 配对差值报告、`budget_comparison.png` 与逐局数据 |

## 快速命令

在本目录内执行（依赖 torch、gymnasium、numpy、matplotlib；实测环境 Python 3.13 +
PyTorch 2.12，CPU 即可运行，GPU 更快）：

```bash
# 从头复现主实验（RTX 3090 约 0.4 小时；结果写入本目录 results/）
python train.py --run-name my_main --iters 3000 --seed 1

# 评估已有模型（1000 局独立测试口径：种子 70000 起）
python evaluate.py --ckpt results/main/best.pt --episodes 1000 --seed 70000

# 重新生成全部图表
python make_ppo_report_figs.py

# 超参数敏感性实验（约 3.5 小时 @ 双 GPU；无 GPU 时加 --devices cpu，耗时显著更长）
python run_sensitivity_experiments.py
python compare_sensitivity.py
```


## 主要结果速览

- 独立测试（默认配置，1000 局，种子 70000–70999，`results/main/best.pt`）：
  棋盘和 **496.9 ± 197.5**，P(≥256) = **82.5%**，P(≥512) = **26.2%**。
- **最佳实验（10 epochs + lr 6.25e-4，其余同默认配置；5 训练种子平均）**：标准预算
  2458 万步下棋盘和 **593.0 ± 18.9**，P(≥256) = **88.9%**，P(≥512) = **43.9%**（模型位于
  `results/sensitivity/epochs10_lr625e4_s*/best.pt`）。复现：
  `python train.py --run-name best --iters 3000 --seed 1 --epochs 10 --lr 6.25e-4`。
- 消融（328 万步 × 3 种子）：去动作掩码使棋盘和 197.4 → 177.1、P(≥256) 归零，
  是唯一有明确信号的设计；γ 与熵系数的扰动在种子噪声内不敏感。
- 超参数敏感性（完整预算 2458 万步，详见 `results/sensitivity/report.md`）：
  基线重跑 500.1 ± 1.7；γ=0.995 与 ε=0.2 为双向甜点，λ=1.0（放弃 bootstrap）损失
  约 100，无动作掩码 −56.1 且种子方差最小，奖励缩放不敏感（优势标准化吸收）；
  **最优配置 10 epochs + lr 6.25e-4 达 593.0 ± 18.9（+92.9）、P(≥512) = 43.9%**，
  五个种子全部高于任何其他配置，交互为正向叠加；lr 1.25e-3 与 15 epochs 两个
  进一步方向均为负（早停，数据归档于 `results/search_partial/`）。
- **预算充分性**（`results/budget_extension/`）：同配置 2 倍预算（4915 万步）× 3 种子
  进一步提升至棋盘和 **657.7 ± 14.0**，P(≥256) = **92.0%**，P(≥512) = **53.0%**，
  三个种子配对差值全部显著为正——2458 万步并非训练充分的上限。
