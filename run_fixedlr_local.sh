#!/bin/bash
# 本地实验：从头训练但全程恒定 lr = 2e-5（对照组）
# 对照对象：三阶段调度的第一阶段末（it 6000，棋盘和约 670）
# 目的：量化"起步就用小学习率"的代价
cd "/home/liang/Documents/seafile/file/phd/博1/上/强化学习/game2048-master/ppo2048/ppo_experiment"
PY=/home/liang/miniconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

echo "[waiting] 等待 ext3x 完成，腾出 GPU 1"
while [ ! -f results/ext3x.done ]; do sleep 60; done

echo "[fixedlr] 从头训练，恒定 lr=2e-5，6000 iters"
for s in 2 3 4; do
  $PY train.py --run-name fixedlr_s$s \
    --iters 6000 --lr 2e-5 --lr-final-ratio 1.0 \
    --epochs 10 --ema-decay 0.999 --seed $s --results-dir results \
    > results/fixedlr_s$s.out 2>&1 &
done
wait

echo "[eval]"
$PY -c "
from eval_gamma_local import eval_ckpt
for s in (2, 3, 4):
    eval_ckpt(f'results/fixedlr_s{s}/best.pt', f'fixedlr_s{s}', 'results/fixedlr_eval')
" 2>&1 | grep -v Warning
echo DONE > results/fixedlr.done
echo "[fixedlr all done]"
