"""同一 PPO 训练轨迹上的新旧验证选模对照，三种子，双 GPU。

新规则 250 次/500 局保存 best.pt；旧规则 25 次/50 局保存 best_legacy.pt。
统一独立终测两套规则选出的权重及固定末尾 EMA，自动生成配对报告。
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from run_value_clip_ablation import (SOURCES, evaluate, now, paired_difference,
                                    train_all, write_json)


BASE = Path(__file__).resolve().parent
CHECKPOINTS = (("old", "best_legacy.pt"), ("new", "best.pt"),
               ("last_ema", "last_ema_0.999.pt"))


def prepare(output, gpus, episodes, test_seed):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    files = SOURCES + ["run_value_clip_ablation.py", "run_validation_ablation.py"]
    hashes = {}
    for filename in files:
        source = BASE / filename
        shutil.copy2(source, snapshot / filename)
        hashes[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    patch = subprocess.run(["git", "diff", "--", *files], cwd=BASE,
                           capture_output=True, text=True, check=True).stdout
    (snapshot / "changes.patch").write_text(patch, encoding="utf-8")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE,
                              capture_output=True, text=True, check=True).stdout.strip()
    commands = []
    for seed, gpu in ((2, gpus[1]), (3, gpus[0]), (4, gpus[1])):
        name = f"validation_s{seed}"
        command = [sys.executable, "-u", "train.py", "--run-name", name,
                   "--results-dir", str(output / "runs"), "--device", "cuda",
                   "--iters", "6000", "--extend-iters", "6000", "--extend-lr", "6e-5",
                   "--extend2-iters", "6000", "--extend2-lr", "2e-5",
                   "--epochs", "10", "--lr", "6.25e-4", "--ema-decay", "0.999",
                   "--eval-every", "250", "--eval-episodes", "500", "--eval-seed", "12345",
                   "--compare-legacy-validation", "--seed", str(seed)]
        commands.append({"name": name, "seed": seed, "gpu": gpu, "command": command})
    gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],
                              capture_output=True, text=True, check=True).stdout
    manifest = {"started": now(), "runner_pid": os.getpid(), "python": sys.executable,
                "git_head": git_head, "source_sha256": hashes, "gpu_info": gpu_info,
                "test_seed": test_seed, "test_episodes": episodes, "value_clip": True,
                "protocols": {"old": {"every": 25, "episodes": 50, "seed": 12345},
                              "new": {"every": 250, "episodes": 500, "seed": 12345}},
                "selection_metric": "board_sum + 250 * p>=256",
                "checkpoint_kinds": [kind for kind, _ in CHECKPOINTS], "runs": commands}
    write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# 验证选模规则对照（运行中）\n\n"
        "三次训练，种子 2/3/4，各 18,000 次迭代、147,456,000 环境步。"
        "沿用三阶段学习率、EMA=0.999、value clipping 开启。\n\n"
        "同一原始/EMA 训练轨迹上独立使用两套规则：\n\n"
        "- 新规则：每 250 次评估固定 500 局，保存 best.pt 与 eval_log.csv。\n"
        "- 旧规则：每 25 次评估固定 50 局，保存 best_legacy.pt 与 eval_legacy_log.csv。\n"
        "- 两套规则均按棋盘和 + 250×P256 选模；均可选原始权重或 EMA。\n"
        "- 另外保存固定末尾 EMA，独立于验证和终测选模。\n\n"
        f"每个候选模型终测 {episodes:,} 局，使用全新种子 "
        f"{test_seed}–{test_seed + episodes - 1}。终测不参与保存 checkpoint。\n\n"
        "本次为配对实验，两套验证同时运行，验证总开销约为单套规则的两倍。"
        "短训练已验证两套规则不会改变原始或 EMA 的参数轨迹。\n\n"
        "status.json 是当前状态；manifest.json 记录命令与源码哈希；source/ 是冻结源码。"
        "检查点在 runs/；训练控制台日志为根目录 *.out。"
        "完成后自动生成 report.md、summary.json、comparison.json 和 evaluation/ 逐局 CSV。\n",
        encoding="utf-8")
    return snapshot, manifest


def report(output, manifest, results, status):
    labels = {"old": "旧规则 25/50", "new": "新规则 250/500", "last_ema": "固定末尾 EMA"}
    episodes = manifest["test_episodes"]
    lines = ["# 同轨迹验证选模规则对照", "",
             f"开始：{manifest['started']}；完成：{now()}。", "",
             "种子 2/3/4 各从头训练一次；每次 6000 次退火 + 6000@6e-5 + 6000@2e-5，"
             "共 147,456,000 环境步，epochs=10，初始 lr=6.25e-4，EMA=0.999，保留 value clipping。", "",
             "每个种子的两套规则使用完全相同的原始/EMA 训练轨迹。新规则每 250 次迭代评估 "
             "500 局；旧规则每 25 次评估 50 局；验证种子均从 12345 开始。"
             "选模目标均为棋盘和 + 250×P256。验证预算按单套规则计算相同，实验旁路同时运行两套。", "",
             f"每模型独立贪心终测 {episodes:,} 局，种子 "
             f"{manifest['test_seed']}–{manifest['test_seed'] + episodes - 1}；"
             "终测不用于重新选 checkpoint。末尾 EMA 为预先指定的固定对照。", "",
             "## 各模型终测", "",
             "| 规则 | 种子 | 选中迭代 | 权重 | 棋盘和 | 得分 | P256 | P512 | P1024 |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for seed in (2, 3, 4):
        for kind, _ in CHECKPOINTS:
            item = results[f"s{seed}_{kind}"][0]
            validation = item.get("validation") or {}
            weight = validation.get("selected_weights", "ema_0.999")
            lines.append(f"| {labels[kind]} | {seed} | {item['checkpoint_iter']} | {weight} | "
                         f"{item['board_sum_mean']:.2f} | {item['score_mean']:.2f} | "
                         f"{item['p>=256']:.2%} | {item['p>=512']:.2%} | {item['p>=1024']:.2%} |")
    aggregates = {}
    lines += ["", "## 三个训练种子汇总", "",
              "均值 ± 样本标准差来自三个训练种子，不是单局结果的标准差。", "",
              "| 规则 | 棋盘和 | 得分 | P512 | P1024 | 终测综合指标 |",
              "|---|---:|---:|---:|---:|---:|"]
    for kind, _ in CHECKPOINTS:
        values = [results[f"s{seed}_{kind}"][0] for seed in (2, 3, 4)]
        item = {key: statistics.mean(value[key] for value in values)
                for key in ("board_sum_mean", "score_mean", "p>=256", "p>=512", "p>=1024",
                            "test_selection_metric")}
        for key in ("board_sum_mean", "score_mean"):
            item[key + "_seed_std"] = statistics.stdev(value[key] for value in values)
        aggregates[kind] = item
        lines.append(f"| {labels[kind]} | {item['board_sum_mean']:.2f} ± "
                     f"{item['board_sum_mean_seed_std']:.2f} | {item['score_mean']:.2f} ± "
                     f"{item['score_mean_seed_std']:.2f} | {item['p>=512']:.2%} | "
                     f"{item['p>=1024']:.2%} | {item['test_selection_metric']:.2f} |")
    comparisons = {}
    lines += ["", "## 逐局配对差值", "",
              "区间为相同测试种子逐局配对 bootstrap 95% CI，2000 次重采样。"
              "它反映测试随机性，不替代训练种子的差异。", "",
              "| 比较 | 训练种子 | 棋盘和差值 [95% CI] | 得分差值 [95% CI] |",
              "|---|---:|---:|---:|"]
    for reference in ("old", "last_ema"):
        for seed in (2, 3, 4):
            delta = {key: paired_difference(results[f"s{seed}_{reference}"][1],
                                            results[f"s{seed}_new"][1], key)
                     for key in ("board_sum", "score")}
            comparisons[f"s{seed}_new_minus_{reference}"] = delta
            cells = []
            for key in ("board_sum", "score"):
                value = delta[key]
                cells.append(f"{value['mean']:+.2f} [{value['ci95'][0]:+.2f}, {value['ci95'][1]:+.2f}]")
            lines.append(f"| 新规则 − {labels[reference]} | {seed} | {cells[0]} | {cells[1]} |")
    mean_gain = aggregates["new"]["board_sum_mean"] - aggregates["old"]["board_sum_mean"]
    primary = [comparisons[f"s{seed}_new_minus_old"]["board_sum"] for seed in (2, 3, 4)]
    if all(item["ci95"][0] > 0 for item in primary):
        verdict = "三个种子均有明确正向的逐局配对证据，支持采用新验证规则。"
    elif all(item["mean"] > 0 for item in primary):
        verdict = "三个种子的棋盘和差值方向均为正，但部分差异仍有测试不确定性。"
    elif mean_gain > 0:
        verdict = "平均棋盘和数值改善，但种子之间效果不一致。"
    else:
        verdict = "本轮未观察到新规则的平均棋盘和改善，需结合综合选模指标及固定末尾 EMA 判断。"
    lines += ["", "## 结论", "", f"新规则 − 旧规则的平均棋盘和：{mean_gain:+.2f}。{verdict}", "",
              "## 选模估计与独立终测", "",
              "差值为保存该 checkpoint 时的验证综合指标减去其终测综合指标。"
              "它同时受有限样本误差与选模影响，不应直接解读成模型训练增益。", "",
              "| 规则 | 种子 | 验证综合指标 | 终测综合指标 | 差值 |",
              "|---|---:|---:|---:|---:|"]
    for seed in (2, 3, 4):
        for kind in ("old", "new"):
            item = results[f"s{seed}_{kind}"][0]
            value = item["validation"]["metric"]
            lines.append(f"| {labels[kind]} | {seed} | {value:.2f} | "
                         f"{item['test_selection_metric']:.2f} | {value - item['test_selection_metric']:+.2f} |")
    lines += ["", "各 checkpoint 的 validation 字段记录实际选模规则、局数及原始/EMA 来源。"
              "训练进程 args 同时包含两套规则开关，不应用 args.eval_episodes 推断旧规则检查点的验证局数。", "",
              "源码与哈希在 source/、manifest.json；每个种子的两套验证曲线在 runs/；"
              "独立终测逐局数据在 evaluation/。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "summary.json", {key: value[0] for key, value in results.items()})
    write_json(output / "aggregate.json", aggregates)
    write_json(output / "comparison.json", comparisons)
    status.update(state="complete", updated=now(), finished=now(),
                  mean_new_minus_old_board_sum=mean_gain, verdict=verdict,
                  report=str(output / "report.md"))
    write_json(output / "status.json", status)
    (output / "DONE").write_text(now() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--test-episodes", type=int, default=10000)
    parser.add_argument("--test-seed", type=int, default=2000000)
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 2 or args.test_episodes < 2:
        parser.error("需要两张 GPU，test-episodes 至少为 2")
    if set(range(12345, 12845)).intersection(range(args.test_seed, args.test_seed + args.test_episodes)):
        parser.error("终测种子不能与验证集重叠")
    output = args.output.resolve()
    snapshot, manifest = prepare(output, gpus, args.test_episodes, args.test_seed)
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
        for run in manifest["runs"]:
            for kind, filename in CHECKPOINTS:
                key = f"s{run['seed']}_{kind}"
                status.update(evaluating=key, updated=now())
                write_json(output / "status.json", status)
                checkpoint = output / "runs" / run["name"] / filename
                item, rows = evaluate(checkpoint, evaluation / key, args.test_episodes,
                                      args.test_seed, f"cuda:{gpus[1]}")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                item["validation"] = saved.get("validation")
                item["test_selection_metric"] = item["board_sum_mean"] + 250 * item["p>=256"]
                write_json(evaluation / f"{key}.json", item)
                results[key] = (item, rows)
        report(output, manifest, results, status)
        print(f"DONE: {output / 'report.md'}", flush=True)
    except BaseException as error:
        status.update(state="failed", error=repr(error), updated=now())
        write_json(output / "status.json", status)
        (output / "FAILED").write_text(repr(error) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
