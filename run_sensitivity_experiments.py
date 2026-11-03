"""PPO 超参数敏感性实验：14 个配置 × 3 训练种子，全部 24,576,000 步从头训练。

单变量设计（OFAT）：每个配置只改动一个超参数，其余与主实验（results/main）完全
一致；baseline 配置即主实验配置，用种子 2、3、4 重跑以便与所有配置做同种子集对比。
结果目录 results/sensitivity/{config}_s{seed}/，完成后写入 done.json；
对比与报告由 compare_sensitivity.py 生成。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).parent

CONFIGS = {
    'baseline': [],
    'gamma099': ['--gamma', '0.99'],
    'gamma0999': ['--gamma', '0.999'],
    'lam09': ['--lam', '0.9'],
    'lam10': ['--lam', '1.0'],
    'clip01': ['--clip', '0.1'],
    'clip03': ['--clip', '0.3'],
    'epochs3': ['--epochs', '3'],
    'epochs10': ['--epochs', '10'],
    'epochs10_lr625e4': ['--epochs', '10', '--lr', '6.25e-4'],
    'combo_lr125e3': ['--epochs', '10', '--lr', '1.25e-3'],
    'combo_epochs15': ['--epochs', '15', '--lr', '6.25e-4'],
    'lr1e4': ['--lr', '1e-4'],
    'lr625e4': ['--lr', '6.25e-4'],
    'ent005': ['--ent-coef', '0.005', '--ent-final', '0.0005'],
    'rew005': ['--reward-scale', '0.05'],
    'nomask': ['--no-mask'],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seeds', nargs='+', type=int, default=[2, 3, 4])
    p.add_argument('--configs', nargs='+', choices=sorted(CONFIGS), default=sorted(CONFIGS))
    p.add_argument('--devices', nargs='+', default=None,
                   help='训练设备列表，轮询分配；缺省自动检测 GPU，无 GPU 时用 cpu')
    p.add_argument('--root', type=Path, default=BASE / 'results/sensitivity')
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.devices:
        devices = args.devices
    else:
        try:
            import torch
            n_gpu = torch.cuda.device_count()
        except ImportError:
            n_gpu = 0
        devices = [f'cuda:{i}' for i in range(n_gpu)] or ['cpu']
    jobs = [(config, seed, devices[i % len(devices)])
            for i, (config, seed) in enumerate((c, s) for c in args.configs for s in args.seeds)]

    def run(job):
        config, seed, device = job
        out = args.root / f'{config}_s{seed}'
        if (out / 'done.json').exists():
            return f'{config} s{seed}: 已完成，复用'
        if out.exists():
            stamp = time.strftime('%Y%m%d-%H%M%S')
            shutil.move(out, out.with_name(out.name + f'.broken-{stamp}'))
            print(f'{config} s{seed}: 发现未完成目录，移至 .broken 后重跑', flush=True)
        cmd = [sys.executable, str(BASE / 'train.py'), '--run-name', f'{config}_s{seed}',
               '--iters', str(args.iters), '--seed', str(seed), '--device', device,
               '--results-dir', str(args.root)] + CONFIGS[config]
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        start = time.perf_counter()
        with (args.root / f'{config}_s{seed}.out').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        seconds = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(f'{config} s{seed} 失败（exit {result.returncode}），查看对应 .out')
        out.joinpath('done.json').write_text(json.dumps(
            dict(seconds=seconds, iters=args.iters, seed=seed, device=device), indent=2))
        return f'{config} s{seed}: 完成，{seconds / 60:.1f} 分钟'

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            print(f'[{n}/{len(jobs)}] {future.result()}', flush=True)


if __name__ == '__main__':
    main()
