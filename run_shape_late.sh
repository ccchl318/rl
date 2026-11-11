#!/bin/bash
# R4: 后期门限分支 GRPO（branch-start-iter=2000），本地 GPU 1，seeds 2/3/4
cd "/home/liang/Documents/seafile/file/phd/博1/上/强化学习/game2048-master/ppo2048/ppo_experiment"
PY=/home/liang/miniconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 2 3 4; do
  $PY train.py --run-name branch_late_s$s --iters 6000 --epochs 10 --lr 6.25e-4 \
    --branch-k 512 --branch-g 8 --branch-h 32 --branch-start-iter 2000 --seed $s \
    --results-dir results > results/branch_late_s$s.out 2>&1 &
done
wait
echo "[eval]"
$PY -c "from eval_gamma_local import eval_ckpt; [eval_ckpt(f'results/branch_late_s{s}/best.pt', f'branch_late_s{s}', 'results/shape_eval') for s in (2,3,4)]" 2>&1 | grep -v Warning
echo DONE > results/branch_late.done
echo "[all done]"
