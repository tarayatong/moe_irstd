"""验证路由稀疏性、推理确定性与训练梯度。"""

import unittest
import torch
from torch import nn
from sfaoe.routing import SpatialNoisyTopkRouter, SpatialSparseMoE


class RoutingTests(unittest.TestCase):
    def test_region_router_normalization_and_eval_determinism(self):
        router = SpatialNoisyTopkRouter(8, 4, 2, patch_size=3).eval()
        x = torch.randn(2, 8, 9, 11)
        weights, indices = router(x)
        second, _ = router(x)
        self.assertEqual(tuple(weights.shape), (2, 4, 9, 11))
        self.assertEqual(tuple(indices.shape), (2, 2, 9, 11))
        torch.testing.assert_close(weights.sum(1), torch.ones(2, 9, 11))
        self.assertTrue(torch.equal((weights > 0).sum(1), torch.full((2, 9, 11), 2)))
        torch.testing.assert_close(weights, second, rtol=0, atol=0)

    def test_full_and_sparse_have_finite_gradients(self):
        for mode in ("full", "sparse"):
            with self.subTest(mode=mode):
                experts = nn.ModuleList(
                    [nn.Conv2d(8, 8, 3, padding=1) for _ in range(4)]
                )
                moe = SpatialSparseMoE(8, 4, 2, 8, experts, input_routing=mode)
                x = torch.randn(2, 8, 8, 8, requires_grad=True)
                output = moe(x)
                loss = output.square().mean() + 0.01 * moe.last_aux_loss
                loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(
                    torch.isfinite(moe.router.topkroute_conv.weight.grad).all()
                )

    def test_legacy_import_is_same_class(self):
        from model.mask import SpatialSparseMoE as LegacyMoE

        self.assertIs(LegacyMoE, SpatialSparseMoE)


if __name__ == "__main__":
    unittest.main()
