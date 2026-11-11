#!/bin/bash
# R1: bonus512 塑形，本地 GPU 1，2x 预算 seeds 2/3/4
cd "/home/liang/Documents/seafile/file/phd/博1/上/强化学习/game2048-master/ppo2048/ppo_experiment"
PY=/home/liang/miniconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 2 3 4; do
  $PY train.py --run-name shape_bonus512_s$s --iters 6000 --epochs 10 --lr 6.25e-4 \
    --shaping bonus512 --bonus512 200 --seed $s --results-dir results > results/shape_bonus512_s$s.out 2>&1 &
done
wait
echo "[eval]"
$PY -c "from eval_gamma_local import eval_ckpt_auto; [eval_ckpt_auto(f'results/shape_bonus512_s{s}/best.pt', f'shape_bonus512_s{s}', 'results/shape_eval') for s in (2,3,4)]" 2>&1 | grep -v Warning
echo DONE > results/shape_local.done
echo "[all done]"
