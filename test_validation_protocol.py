"""验证双选模规则彼此独立，并且旁路评估不改变原始训练/EMA 轨迹。"""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from train import parse_args


class ValidationDefaultsTests(unittest.TestCase):
    def test_defaults(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = parse_args()
        self.assertEqual((args.eval_every, args.eval_episodes, args.eval_seed), (250, 500, 12345))
        self.assertFalse(args.compare_legacy_validation)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class ValidationTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ppo_validation_check_")
        cls.root = Path(cls.temporary.name)
        cls.saved = {}
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        for name, every, episodes, dual in (("new", 13, 16, False),
                                            ("dual", 13, 16, True),
                                            ("old", 25, 50, False)):
            command = [sys.executable, "train.py", "--run-name", name, "--results-dir", str(cls.root),
                       "--iters", "26", "--n-envs", "8", "--steps", "16", "--hidden", "16",
                       "--epochs", "2", "--minibatch-size", "64", "--device", "cuda",
                       "--seed", "17", "--lr", "6.25e-4", "--ema-decay", "0.9",
                       "--eval-every", str(every), "--eval-episodes", str(episodes)]
            if dual:
                command.append("--compare-legacy-validation")
            result = subprocess.run(command, cwd=Path(__file__).parent, env=environment,
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            cls.saved[name] = {filename: torch.load(cls.root / name / filename,
                                                   map_location="cpu", weights_only=False)
                               for filename in ["last.pt", "last_ema_0.9.pt", "best.pt"]}
        cls.saved["dual"]["best_legacy.pt"] = torch.load(cls.root / "dual" / "best_legacy.pt",
                                                       map_location="cpu", weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assert_weights_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_training_and_ema_unchanged_by_validation(self):
        for filename in ("last.pt", "last_ema_0.9.pt"):
            for mode in ("dual", "old"):
                with self.subTest(checkpoint=filename, mode=mode):
                    self.assert_weights_equal(self.saved["new"][filename]["model"],
                                              self.saved[mode][filename]["model"])

    def test_independent_best_matches_each_standalone_protocol(self):
        for standalone, dual_name in (("new", "best.pt"), ("old", "best_legacy.pt")):
            self.assertEqual(self.saved[standalone]["best.pt"]["it"], self.saved["dual"][dual_name]["it"])
            self.assert_weights_equal(self.saved[standalone]["best.pt"]["model"],
                                      self.saved["dual"][dual_name]["model"])

    def test_schedules_and_selected_checkpoint_metadata(self):
        for log_name, expected_iters, checkpoint, label, count in (
                ("eval_log.csv", [13, 26], "best.pt", "main", 16),
                ("eval_legacy_log.csv", [25, 26], "best_legacy.pt", "legacy", 50)):
            with (self.root / "dual" / log_name).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["iter"]) for row in rows], expected_iters)
            self.assertTrue(all(int(row["eval_episodes"]) == count for row in rows))
            validation = self.saved["dual"][checkpoint]["validation"]
            self.assertEqual(validation["protocol"], label)
            self.assertEqual(validation["episodes"], count)
            self.assertAlmostEqual(validation["metric"], max(float(row["selection_metric"]) for row in rows), places=5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
