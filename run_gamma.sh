#!/bin/bash
# gamma 实验矩阵（本地 GPU 0）：G1=g0.997 | G2=g0.998 | G3=g0.998+T256
cd "/home/liang/Documents/seafile/file/phd/博1/上/强化学习/game2048-master/ppo2048/ppo_experiment"
PY=/home/liang/miniconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "[wave1] G1: gamma=0.997, 64x128x6000, seeds 2/3/4"
for s in 2 3 4; do
  $PY train.py --run-name gamma_g997_s$s --iters 6000 --epochs 10 --lr 6.25e-4 \
    --gamma 0.997 --seed $s --results-dir results > results/gamma_g997_s$s.out 2>&1 &
done
wait

echo "[wave2] G2: gamma=0.998 T128 + G3: gamma=0.998 T256 (32 envs)"
for s in 2 3 4; do
  $PY train.py --run-name gamma_g998_s$s --iters 6000 --epochs 10 --lr 6.25e-4 \
    --gamma 0.998 --seed $s --results-dir results > results/gamma_g998_s$s.out 2>&1 &
done
for s in 2 3 4; do
  $PY train.py --run-name gamma_g998_t256_s$s --iters 6000 --epochs 10 --lr 6.25e-4 \
    --gamma 0.998 --n-envs 32 --steps 256 --seed $s --results-dir results > results/gamma_g998_t256_s$s.out 2>&1 &
done
wait

echo "[eval] 9 models x 1000 episodes"
$PY eval_gamma.py
echo DONE > results/gamma_experiment.done
echo "[all done]"
