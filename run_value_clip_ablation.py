"""本地双 GPU 单变量消融：三阶段 PPO，value clipping on/off，各三种子。

自动冻结源码、运行训练、评测 best 与末尾 EMA，并生成逐局数据和配对报告。
输出目录必须不存在，以防覆盖实验。status.json 可用于查看进度。
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np


BASE = Path(__file__).resolve().parent
SOURCES = ["train.py", "ppo.py", "gpu_graph.py", "game2048_env.py",
           "game2048_env_fast.py", "afterstate.py", "branch_adv.py"]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def train_progress(log_path):
    if not log_path.exists():
        return {}
    with log_path.open("rb") as handle:
        handle.seek(max(0, log_path.stat().st_size - 6000))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = re.findall(r"^it\s+(\d+)\s+\|.*?ips\s+([\d.]+)", tail, re.MULTILINE)
    if not matches:
        return {}
    iteration, ips = matches[-1]
    ips = float(ips)
    return {"iter": int(iteration), "total_iters": 18000, "steps_per_second": ips,
            "train_minutes_remaining": round((18000 - int(iteration)) * 8192 / ips / 60, 1)}


def train_all(output, snapshot, manifest, status):
    processes = []
    for run in manifest["runs"]:
        log_path = output / f"{run['name']}.out"
        handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(run["gpu"]),
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           PYTHONUNBUFFERED="1")
        process = subprocess.Popen(run["command"], cwd=snapshot, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        handle.close()
        status["runs"][run["name"]] = {
            "pid": process.pid, "gpu": run["gpu"], "state": "training", "started": now()}
        processes.append((run, process, log_path))
        print(f"Started {run['name']} on GPU {run['gpu']}, pid={process.pid}", flush=True)
    write_json(output / "status.json", status)
    next_print = 0.0
    while True:
        running = 0
        for run, process, log_path in processes:
            item = status["runs"][run["name"]]
            item.update(train_progress(log_path))
            code = process.poll()
            if code is None:
                running += 1
            elif item["state"] == "training":
                item.update(state="trained" if code == 0 else "failed",
                            returncode=code, finished=now())
                print(f"Finished {run['name']}: returncode={code}", flush=True)
        status["updated"] = now()
        write_json(output / "status.json", status)
        if time.monotonic() >= next_print:
            print(json.dumps({name: {k: item[k] for k in
                                    ("state", "iter", "train_minutes_remaining") if k in item}
                              for name, item in status["runs"].items()}, ensure_ascii=False), flush=True)
            next_print = time.monotonic() + 60
        if not running:
            break
        time.sleep(10)
    failed = [name for name, item in status["runs"].items() if item["state"] == "failed"]
    if failed:
        raise RuntimeError(f"训练失败，请查看相应 .out 日志：{failed}")


def wilson(successes, count):
    z = 1.96
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - half, center + half]


def evaluate(checkpoint, target, episodes, seed, device):
    import torch
    from ppo import ActorCritic
    from game2048_env_fast import FastGame2048Batch

    started = time.monotonic()
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ActorCritic(hidden=saved["args"]["hidden"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    rows = []
    with torch.inference_mode():
        for offset in range(0, episodes, 500):
            count = min(500, episodes - offset)
            env = FastGame2048Batch(count, seeds=[seed + offset + i for i in range(count)])
            alive = np.ones(count, dtype=bool)
            while alive.any():
                indices = np.flatnonzero(alive)
                obs = torch.from_numpy(env.obs[indices]).to(device)
                mask = torch.from_numpy(env.mask[indices]).to(device)
                actions = np.zeros(count, dtype=np.int64)
                actions[indices] = model.policy_logits(obs, mask).argmax(-1).cpu().numpy()
                env.step(actions)
                for index, stat in env.pop_finished():
                    if alive[index]:
                        alive[index] = False
                        score, tile, board_sum, length = stat
                        rows.append({"seed": seed + offset + index, "score": score,
                                     "board_sum": board_sum, "max_tile": tile, "length": length})
    rows.sort(key=lambda row: row["seed"])
    assert len(rows) == episodes and len({row["seed"] for row in rows}) == episodes
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint": str(checkpoint), "checkpoint_iter": saved["it"],
               "episodes": episodes, "seed_start": seed, "seed_end": seed + episodes - 1,
               "no_value_clip": saved["args"]["no_value_clip"]}
    for key in ("score", "board_sum", "length"):
        values = np.array([row[key] for row in rows])
        summary[key + "_mean"] = float(values.mean())
        summary[key + "_std"] = float(values.std())
        summary[key + "_se"] = float(values.std(ddof=1) / math.sqrt(episodes))
    for threshold in (128, 256, 512, 1024):
        successes = sum(row["max_tile"] >= threshold for row in rows)
        summary[f"p>={threshold}"] = successes / episodes
        summary[f"p>={threshold}_ci95"] = wilson(successes, episodes)
    summary["seconds"] = round(time.monotonic() - started, 1)
    write_json(target.with_suffix(".json"), summary)
    print(f"Evaluated {target.name}: sum={summary['board_sum_mean']:.2f}, "
          f"score={summary['score_mean']:.2f}, p512={summary['p>=512']:.3%}", flush=True)
    return summary, rows


def paired_difference(a, b, key):
    assert [row["seed"] for row in a] == [row["seed"] for row in b]
    delta = np.array([right[key] - left[key] for left, right in zip(a, b)], dtype=np.float64)
    rng = np.random.default_rng(20261116)
    bootstrap = []
    for _ in range(20):
        indices = rng.integers(0, len(delta), size=(100, len(delta)))
        bootstrap.extend(delta[indices].mean(axis=1).tolist())
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    return {"mean": float(delta.mean()), "ci95": [float(lo), float(hi)],
            "bootstrap_samples": 2000}


def write_report(output, manifest, results, episodes, status):
    lines = ["# Value clipping 单变量消融", "",
             f"开始：{manifest['started']}；完成：{now()}。", "",
             "两组从头训练，均为 6000 次退火 + 6000@6e-5 + 6000@2e-5，"
             "每次 8192 步（共 147,456,000 步）；epochs=10，初始 lr=6.25e-4，EMA=0.999。",
             "唯一训练处理变量为 --no-value-clip；策略裁剪、梯度范数裁剪、奖励与选模规则相同。", "",
             f"每个 checkpoint 独立贪心评测 {episodes:,} 局，种子 "
             f"{manifest['test_seed']}–{manifest['test_seed'] + episodes - 1}。",
             "best 仍按历史协议的固定 50 局验证指标选取；末尾 EMA 单独评测，"
             "不使用终测结果再次挑选 checkpoint。", "",
             "| checkpoint | 种子 | 裁剪 | 棋盘和 | 得分 | P256 | P512 | P1024 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for kind in ("best", "last_ema"):
        for seed in (2, 3, 4):
            for arm in ("clip", "noclip"):
                item = results[f"{arm}_s{seed}_{kind}"][0]
                lines.append(f"| {kind} | {seed} | {arm} | {item['board_sum_mean']:.2f} | "
                             f"{item['score_mean']:.2f} | {item['p>=256']:.2%} | "
                             f"{item['p>=512']:.2%} | {item['p>=1024']:.2%} |")
    comparisons = {}
    lines += ["", "## 跨训练种子汇总", "",
              "均值 ± 样本标准差来自三个训练种子，不是单局分数的标准差。", "",
              "| checkpoint | 裁剪 | 棋盘和 | 得分 | P512 |",
              "|---|---|---:|---:|---:|"]
    for kind in ("best", "last_ema"):
        for arm in ("clip", "noclip"):
            summaries = [results[f"{arm}_s{seed}_{kind}"][0] for seed in (2, 3, 4)]
            sums = [row["board_sum_mean"] for row in summaries]
            scores = [row["score_mean"] for row in summaries]
            lines.append(f"| {kind} | {arm} | {statistics.mean(sums):.2f} ± "
                         f"{statistics.stdev(sums):.2f} | {statistics.mean(scores):.2f} ± "
                         f"{statistics.stdev(scores):.2f} | "
                         f"{statistics.mean(row['p>=512'] for row in summaries):.2%} |")
    lines += ["", "## 逐局配对差值：noclip − clip", "",
              "置信区间为同测试种子逐局配对 bootstrap，反映测试随机性，不能替代训练种子的差异。", "",
              "| checkpoint | 训练种子 | 棋盘和差值 [95% CI] | 得分差值 [95% CI] |",
              "|---|---:|---:|---:|"]
    for kind in ("best", "last_ema"):
        for seed in (2, 3, 4):
            clipped = results[f"clip_s{seed}_{kind}"][1]
            unclipped = results[f"noclip_s{seed}_{kind}"][1]
            differences = {key: paired_difference(clipped, unclipped, key)
                           for key in ("board_sum", "score")}
            comparisons[f"s{seed}_{kind}"] = differences
            text = []
            for key in ("board_sum", "score"):
                item = differences[key]
                text.append(f"{item['mean']:+.2f} [{item['ci95'][0]:+.2f}, {item['ci95'][1]:+.2f}]")
            lines.append(f"| {kind} | {seed} | {text[0]} | {text[1]} |")
    primary = [comparisons[f"s{seed}_best"]["board_sum"] for seed in (2, 3, 4)]
    mean_gain = statistics.mean(item["mean"] for item in primary)
    if all(item["ci95"][0] > 0 for item in primary):
        verdict = "三个训练种子的 best 棋盘和均有正向配对证据，支持关闭 value clipping。"
    elif all(item["mean"] < 0 for item in primary):
        verdict = "三个训练种子的 best 棋盘和均下降，不支持采用关闭 value clipping。"
    else:
        verdict = "没有得到三个训练种子一致且明确的正向证据，需结合末尾 EMA 和训练方差判断。"
    lines += ["", "## 判定", "", f"平均 best 棋盘和差值：{mean_gain:+.2f}。{verdict}", "",
              "## 训练诊断（最后 3000 次迭代均值）", "",
              "v_clip_frac 是裁剪分支屏蔽价值梯度的样本比例；v_mse 是未裁剪的平方误差。"
              "rollout_explained_var 使用本批 GAE 目标，不是独立 Monte Carlo 真值。", "",
              "| 运行 | v_clip_frac | v_mse | rollout_explained_var |",
              "|---|---:|---:|---:|"]
    for run in manifest["runs"]:
        with (output / "runs" / run["name"] / "train_log.csv").open() as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["iter"]) > 15000]
        metrics = [statistics.mean(float(row[key]) for row in rows)
                   for key in ("v_clip_frac", "v_mse", "rollout_explained_var")]
        lines.append(f"| {run['name']} | {metrics[0]:.3%} | {metrics[1]:.2f} | {metrics[2]:.4f} |")
    lines += ["", "源码快照与哈希在 source/ 和 manifest.json；完整进度在 status.json；"
              "每次训练日志为根目录 *.out；检查点在 runs/；逐局数据在 evaluation/。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "comparison.json", comparisons)
    write_json(output / "summary.json", {key: value[0] for key, value in results.items()})
    status.update(state="complete", finished=now(), mean_best_board_sum_gain=mean_gain,
                  verdict=verdict, report=str(output / "report.md"))
    write_json(output / "status.json", status)
    (output / "DONE").write_text(now() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--test-episodes", type=int, default=10000)
    parser.add_argument("--test-seed", type=int, default=1000000)
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 2 or args.test_episodes < 2:
        parser.error("需要两张 GPU，并且 test-episodes 至少为 2")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    hashes = {}
    for filename in SOURCES:
        source = BASE / filename
        shutil.copy2(source, snapshot / filename)
        hashes[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    patch = subprocess.run(["git", "diff", "--", *SOURCES], cwd=BASE,
                           capture_output=True, text=True, check=True).stdout
    (snapshot / "changes.patch").write_text(patch, encoding="utf-8")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE,
                              capture_output=True, text=True, check=True).stdout.strip()
    commands = []
    # 每卡三个进程；s2/s3 的两组在同一卡，s4 的两组分布在两张同型号卡。
    assignments = [(2, "clip", gpus[1]), (2, "noclip", gpus[1]),
                   (3, "clip", gpus[0]), (3, "noclip", gpus[0]),
                   (4, "clip", gpus[0]), (4, "noclip", gpus[1])]
    for seed, arm, gpu in assignments:
        name = f"{arm}_s{seed}"
        command = [sys.executable, "-u", "train.py", "--run-name", name,
                   "--results-dir", str(output / "runs"), "--device", "cuda",
                   "--iters", "6000", "--extend-iters", "6000", "--extend-lr", "6e-5",
                   "--extend2-iters", "6000", "--extend2-lr", "2e-5",
                   "--epochs", "10", "--lr", "6.25e-4", "--ema-decay", "0.999",
                   "--eval-every", "25", "--eval-episodes", "50", "--eval-seed", "12345",
                   "--seed", str(seed)]
        if arm == "noclip":
            command.append("--no-value-clip")
        commands.append({"name": name, "seed": seed, "arm": arm, "gpu": gpu, "command": command})
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],
                              capture_output=True, text=True, check=True).stdout
    manifest = {"started": now(), "runner_pid": os.getpid(), "python": sys.executable,
                "git_head": git_head, "source_sha256": hashes, "gpu_info": gpu_info,
                "test_seed": args.test_seed, "test_episodes": args.test_episodes,
                "checkpoint_kinds": ["best", "last_ema"], "runs": commands}
    write_json(output / "manifest.json", manifest)
    status = {"state": "training", "started": now(), "runner_pid": os.getpid(), "runs": {}}
    try:
        train_all(output, snapshot, manifest, status)
        status.update(state="evaluating", updated=now())
        write_json(output / "status.json", status)
        sys.path.insert(0, str(snapshot))
        import torch
        torch.set_num_threads(2)
        evaluation = output / "evaluation"
        evaluation.mkdir()
        results = {}
        for run in commands:
            for kind, filename in (("best", "best.pt"), ("last_ema", "last_ema_0.999.pt")):
                key = f"{run['name']}_{kind}"
                status.update(evaluating=key, updated=now())
                write_json(output / "status.json", status)
                results[key] = evaluate(output / "runs" / run["name"] / filename,
                                        evaluation / key, args.test_episodes, args.test_seed,
                                        f"cuda:{gpus[1]}")
        write_report(output, manifest, results, args.test_episodes, status)
        print(f"DONE: {output / 'report.md'}", flush=True)
    except BaseException as error:
        status.update(state="failed", error=repr(error), updated=now())
        write_json(output / "status.json", status)
        (output / "FAILED").write_text(repr(error) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
