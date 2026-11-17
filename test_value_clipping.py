"""检查 value clipping 的梯度语义，以及 CUDA Graph 与普通更新的一致性。"""
import unittest

import torch

from gpu_graph import GraphedUpdater
from ppo import ActorCritic, ppo_update


class ValueClippingTests(unittest.TestCase):
    def run_update(self, value_clip=True, graph=False, explicit=True):
        device = "cuda" if graph else "cpu"
        model = ActorCritic(hidden=16).to(device)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.v_net[-1].bias.fill_(101.0)
        batch = {
            "obs": torch.zeros(8, 144),
            "mask": torch.ones(8, 4, dtype=torch.bool),
            "act": torch.zeros(8, dtype=torch.long),
            "logp": torch.full((8,), -torch.log(torch.tensor(4.0)).item()),
            "adv": torch.zeros(8),
            "ret": torch.full((8,), 110.0),
            "val": torch.full((8,), 100.0),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        if graph:
            updater = GraphedUpdater(model, optimizer, 8, 8, 1.0, 0.2,
                                     device, epochs=1, max_grad_norm=1e6,
                                     value_clip=value_clip)
            stats = updater.update(batch, 0.0)
        else:
            kwargs = {"value_clip": value_clip} if explicit else {}
            stats = ppo_update(model, optimizer, batch, 0.2, 1, 8, 1.0,
                               0.0, max_grad_norm=1e6, **kwargs)
        return model.v_net[-1].bias.item(), stats

    def test_clipped_branch_blocks_further_correction(self):
        prediction, stats = self.run_update(True)
        self.assertAlmostEqual(prediction, 101.0, places=4)
        self.assertAlmostEqual(stats["v_loss"], 48.02, places=3)
        self.assertEqual(stats["v_clip_frac"], 1.0)

    def test_disabling_value_clip_restores_correction(self):
        prediction, stats = self.run_update(False)
        self.assertAlmostEqual(prediction, 101.9, places=4)
        self.assertAlmostEqual(stats["v_loss"], 40.5, places=4)
        self.assertAlmostEqual(stats["v_mse"], 81.0, places=4)
        self.assertEqual(stats["v_clip_frac"], 0.0)

    def test_default_preserves_clipping(self):
        self.assertEqual(self.run_update(explicit=False), self.run_update(True))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_graph_matches_regular_update(self):
        for enabled in (True, False):
            with self.subTest(value_clip=enabled):
                expected, expected_stats = self.run_update(enabled)
                actual, actual_stats = self.run_update(enabled, graph=True)
                self.assertAlmostEqual(actual, expected, places=4)
                for key in expected_stats:
                    self.assertAlmostEqual(actual_stats[key], expected_stats[key], places=3)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
