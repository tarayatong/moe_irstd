"""参数优先级回归测试，不创建实验目录、不加载 GPU。"""

import unittest
from sfaoe.config import parse_args


class ConfigurationTests(unittest.TestCase):
    def test_original_effective_defaults(self):
        args = parse_args([], initialize_run=False)
        self.assertEqual(
            (args.dataset, args.epochs, args.input_routing),
            ("NUAA-SIRST", 1000, "sparse"),
        )
        self.assertEqual(args.moe_stages, [True] * 4)
        self.assertIsNone(args.save_dir)

    def test_explicit_options_are_not_overwritten(self):
        args = parse_args(
            [
                "--dataset",
                "IRSTD",
                "--epochs",
                "300",
                "--base_size",
                "512",
                "--crop_size",
                "512",
                "--patch_size",
                "1,2,4,8",
                "--moe_stages",
                "0,0,1,1",
                "--input_routing",
                "full",
                "--save_routing",
                "none",
                "--routing_epochs",
                "",
                "--use_compile",
                "False",
                "--deep_supervision",
                "False",
            ],
            initialize_run=False,
        )
        self.assertEqual(args.dataset, "IRSTD")
        self.assertEqual(args.epochs, 300)
        self.assertEqual((args.base_size, args.crop_size), (512, 512))
        self.assertEqual(args.patch_size, [1, 2, 4, 8])
        self.assertEqual(args.moe_stages, [False, False, True, True])
        self.assertEqual(args.input_routing, "full")
        self.assertEqual(args.routing_epochs, set())
        self.assertFalse(args.use_compile)
        self.assertFalse(args.deep_supervision)

    def test_invalid_granularity_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--patch_size", "1,2"], initialize_run=False)


if __name__ == "__main__":
    unittest.main()
