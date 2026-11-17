#!/bin/bash
# 本地实验：把第三阶段（2e-5）从 6000 迭代延长到 12000
# 目的：验证 ultrafine 的第三阶段是否被截断（s4 在该阶段曾暴涨 +137）
# 前 18000 迭代与 ultrafine 完全相同（同种子同调度），因此可直接配对比较
cd "/home/liang/Documents/seafile/file/phd/博1/上/强化学习/game2048-master/ppo2048/ppo_experiment"
PY=/home/liang/miniconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

echo "[ext3x] 6000 退火 + 6000@6e-5 + 12000@2e-5 = 24000 iters (196.6M 步)"
for s in 2 3 4; do
  $PY train.py --run-name ext3x_s$s \
    --iters 6000 --extend-iters 6000 --extend-lr 6e-5 \
    --extend2-iters 12000 --extend2-lr 2e-5 \
    --epochs 10 --lr 6.25e-4 --ema-decay 0.999 --seed $s --results-dir results \
    > results/ext3x_s$s.out 2>&1 &
done
wait

echo "[eval]"
$PY -c "
from eval_gamma_local import eval_ckpt
for s in (2, 3, 4):
    eval_ckpt(f'results/ext3x_s{s}/best.pt', f'ext3x_s{s}', 'results/ext3x_eval')
" 2>&1 | grep -v Warning
echo DONE > results/ext3x.done
echo "[ext3x all done]"
