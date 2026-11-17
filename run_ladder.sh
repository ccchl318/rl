#!/bin/bash
# 本地 GPU 0：对照"组合实验"的最后一段 lr
#   组合（远程）: [6000退火][6000@6e-5][6000@2e-5][3000@5e-6]  -> 755.3
#   本实验      : [6000退火][6000@6e-5][6000@2e-5][3000@1e-5]  <- 只改最后一段
# 目的：5e-6 太小（无收益）、2e-5 仍有效，测 1e-5 是否为甜点
cd "/home/liang/Documents/seafile/file/phd/博1/上/强化学习/game2048-master/ppo2048/ppo_experiment"
PY=/home/liang/miniconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

echo "[ladder] 21000 iters，最后 3000 @ 1e-5"
for s in 2 3 4; do
  $PY train.py --run-name ladder_s$s \
    --iters 6000 --extend-iters 6000 --extend-lr 6e-5 \
    --extend2-iters 6000 --extend2-lr 2e-5 \
    --extend3-iters 3000 --extend3-lr 1e-5 \
    --epochs 10 --lr 6.25e-4 --ema-decay 0.999 --seed $s --results-dir results \
    > results/ladder_s$s.out 2>&1 &
done
wait

echo "[eval]"
$PY -c "
from eval_gamma_local import eval_ckpt
for s in (2, 3, 4):
    eval_ckpt(f'results/ladder_s{s}/best.pt', f'ladder_s{s}', 'results/ladder_eval')
" 2>&1 | grep -v Warning
echo DONE > results/ladder.done
echo "[ladder all done]"
