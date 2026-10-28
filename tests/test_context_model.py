"""Functional checks for the final three-expert sparse model."""

import unittest
import torch
from torch import nn
from sfaoe.config import parse_args
from sfaoe.factory import build_model
from sfaoe.experts import ContextWaveletConv2d, BasicRFB_a
from sfaoe.routing import SpatialSparseMoE
from sfaoe.checkpoints import clean_context_state_dict
from model.load_param_data import load_param


class ContextModelTests(unittest.TestCase):
    def test_default_model_has_fifteen_three_expert_sparse_routers(self):
        args = parse_args([], initialize_run=False)
        filters, blocks = load_param(args.channel_size, args.backbone)
        model = build_model(args, filters, blocks)
        routers = [m for m in model.modules() if isinstance(m, SpatialSparseMoE)]
        self.assertEqual(len(routers), 15)
        for router in routers:
            self.assertEqual((router.num_experts, router.top_k, router.input_routing), (3, 2, "sparse"))
            self.assertIsInstance(router.experts[0], ContextWaveletConv2d)
            self.assertIsInstance(router.experts[1], BasicRFB_a)
            self.assertIsInstance(router.experts[2], nn.Identity)
            self.assertEqual(router.experts[0].wt_levels, 2)
        self.assertFalse(any("hconv" in key or "llconv" in key for key in model.state_dict()))
        model.eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 64))
        self.assertTrue(outputs)
        self.assertEqual(tuple(outputs[-1].shape), (1, 1, 64, 64))
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs))

    def test_context_odd_shape_gradient_and_sample_independence(self):
        expert = ContextWaveletConv2d(4, 4, wt_levels=2).eval()
        x = torch.randn(2, 4, 15, 17, requires_grad=True)
        y = expert(x)
        self.assertEqual(y.shape, x.shape)
        torch.testing.assert_close(y[0:1], expert(x[0:1]))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for conv in expert.wavelet_convs:
            self.assertTrue(torch.isfinite(conv.weight.grad).all())
            self.assertGreater(conv.weight.grad.abs().sum().item(), 0)

    def test_sparse_router_really_masks_each_expert_input(self):
        class Capture(nn.Module):
            def forward(self, x):
                self.input = x.detach().clone()
                return x
        experts = nn.ModuleList([Capture() for _ in range(3)])
        router = SpatialSparseMoE(4, 3, 2, 4, experts, noise_scale=0).eval()
        with torch.no_grad():
            router.router.topkroute_conv.weight.zero_()
            router.router.topkroute_conv.bias.copy_(torch.tensor([3., 2., 1.]))
        router(torch.ones(1, 4, 8, 8))
        self.assertTrue(torch.equal(experts[0].input, torch.ones(1, 4, 8, 8)))
        self.assertFalse(hasattr(experts[2], "input"))

    def test_old_context_dead_keys_are_removed_but_four_experts_rejected(self):
        prefix = "node_list.0.1.1.mixer.m.attn.spacial_moe."
        active = prefix + "experts.0.wavelet_convs.0.weight"
        dead = prefix + "experts.0.hconv.0.offset.weight"
        cleaned = clean_context_state_dict({active: torch.ones(4), dead: torch.ones(3)})
        self.assertEqual(list(cleaned), [active])
        with self.assertRaises(ValueError):
            clean_context_state_dict({prefix + "router.topkroute_conv.weight": torch.ones(4, 8, 1, 1)})

    def test_four_way_topk_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--top_k", "4"], initialize_run=False)


if __name__ == "__main__":
    unittest.main()
