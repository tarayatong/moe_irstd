"""用小模型验证训练器实例配置与快照异常恢复，不启动实际训练实验。"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from sfaoe.training import Trainer
from sfaoe.routing import SpatialSparseMoE


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = SpatialSparseMoE(4, 4, 2, 4)

    def forward(self, x):
        return [self.moe(x)]


class TrainingTests(unittest.TestCase):
    def make_trainer(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(aux_loss_weight=0.01, train_batch_size=2)
        trainer.model = TinyModel()
        trainer.train_data = [(torch.randn(2, 4, 8, 8), torch.zeros(2, 4, 8, 8))]
        trainer.test_data = trainer.train_data
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.lossfunc = lambda outputs, labels: (
            (outputs[-1] - labels).square().mean()
        )
        return trainer

    def test_training_uses_instance_args_without_module_global(self):
        trainer = self.make_trainer()
        before = trainer.model.moe.router.topkroute_conv.weight.detach().clone()
        with (
            patch.object(torch.Tensor, "cuda", lambda tensor: tensor),
            patch("torch.cuda.synchronize"),
        ):
            trainer.training(0)
        self.assertTrue(torch.isfinite(torch.tensor(trainer.train_loss)))
        self.assertFalse(
            torch.equal(before, trainer.model.moe.router.topkroute_conv.weight)
        )

    def test_snapshot_preserves_eval_mode_and_exports_expected_metadata(self):
        trainer = self.make_trainer()
        trainer.model.eval()
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as directory:
            with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
                trainer.save_routing_snapshot(3, directory, mode="full")
            payload = torch.load(
                Path(directory) / "routing_epoch_3.pt", weights_only=True
            )
        self.assertFalse(trainer.model.training)
        self.assertIsNone(trainer.model.moe.record_routing)
        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["meta"]["mode"], "full")
        self.assertEqual(len(payload["modules"]), 1)
        self.assertIn("input", payload)

    def test_snapshot_restores_mode_after_loader_failure(self):
        trainer = self.make_trainer()
        trainer.model.train()
        trainer.test_data = []
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as directory:
            with self.assertRaises(StopIteration):
                trainer.save_routing_snapshot(0, directory)
        self.assertTrue(trainer.model.training)
        self.assertIsNone(trainer.model.moe.record_routing)


if __name__ == "__main__":
    unittest.main()
