"""兼容原训练入口：python train_spar.py [options]。"""

from sfaoe.config import parse_args

if __name__ == "__main__":
    # 先处理 --help，再加载训练/GPU 依赖。
    args = parse_args()
    from sfaoe.training import main

    main(args)
else:
    # 外部回放脚本仍可使用 train_spar.Trainer / train_spar.main。
    from sfaoe.training import Trainer, main
