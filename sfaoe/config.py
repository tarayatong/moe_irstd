"""训练参数：解析无文件副作用，运行目录在显式初始化时创建。"""

import argparse


def _boolean(value):
    """避免 bool('False') 为 True 的命令行陷阱。"""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false, yes/no or 1/0")


def _integers(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser():
    """原有效默认值作为默认配置，显式命令行参数拥有优先权。"""
    parser = argparse.ArgumentParser(
        description="SFAoE infrared small target detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, help="model name: DNANet", default="DNANet"
    )
    parser.add_argument(
        "--channel_size", type=str, default="three", help="one,  two,  three,  four"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        help="vgg10, resnet_10,  resnet_18,  resnet_34 ",
        default="resnet_18",
    )
    parser.add_argument(
        "--deep_supervision",
        help="True or False (model==DNANet)",
        type=_boolean,
        default=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name:  NUDT-SIRST, NUAA-SIRST, NUST-SIRST",
        default="NUAA-SIRST",
    )
    parser.add_argument(
        "--mode", type=str, help="mode name:  TXT, Ratio", default="TXT"
    )
    parser.add_argument(
        "--test_size", type=float, default="0.5", help="when mode==Ratio"
    )
    parser.add_argument("--root", type=str, default="dataset/")
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument(
        "--split_method",
        type=str,
        help="50_50, 10000_100(for NUST-SIRST)",
        default="50_50",
    )
    parser.add_argument(
        "--workers", type=int, default=0, metavar="N", help="dataloader threads"
    )
    parser.add_argument(
        "--in_channels", type=int, default=3, help="in_channel=3 for pre-process"
    )
    parser.add_argument("--base_size", type=int, help="base image size", default=256)
    parser.add_argument("--crop_size", type=int, help="crop image size", default=256)
    parser.add_argument(
        "--epochs",
        type=int,
        metavar="N",
        help="number of epochs to train",
        default=1000,
    )
    parser.add_argument(
        "--start_epoch", type=int, default=0, metavar="N", help="start epochs"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        metavar="N",
        help="input batch size for                         training",
        default=4,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        metavar="N",
        help="input batch size for                         testing",
        default=4,
    )
    parser.add_argument(
        "--min_lr", default=1e-05, type=float, help="minimum learning rate"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adagrad", help=" Adam, Adagrad"
    )
    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        choices=["CosineAnnealingLR", "ReduceLROnPlateau"],
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, metavar="LR", help="learning rate"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Training with GPUs, you can specify 1,3 for example.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss Function. Choose from 'SoftIoU', 'weighted_l2' or 'combined_loss'",
        default="spar_iou_loss",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="binary",
        help="Label Type. Choose From 'binary' or 'gaussian'.",
    )
    parser.add_argument(
        "--block", type=str, default="Res_CBAM", help="Res_CBAM or Shuffle"
    )
    parser.add_argument(
        "--use_compile",
        help="whether to use torch.compile",
        type=_boolean,
        default=True,
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile optimization mode",
    )
    parser.add_argument(
        "--arch_lr",
        type=float,
        default=0.05,
        help="learning rate for alpha and beta in architect searching process",
    )
    parser.add_argument(
        "--arch_weight_decay", type=float, default=0.001, metavar="M", help="w-decay"
    )
    parser.add_argument(
        "--moe_stages",
        type=str,
        help="which stages use MoE, e.g. 1,1,1,1 for all, 0,0,1,1 for low-res only",
        default="1,1,1,1",
    )
    parser.add_argument(
        "--dilations",
        type=str,
        help="dilation rates for BasicRFB_a branches, e.g. 1,2,2,3",
        default="1,2,2,3",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        help="noise scale factor alpha for SpatialNoisyTopkRouter",
        default=0.2,
    )
    parser.add_argument(
        "--patch_size",
        type=str,
        help='MoE routing granularity. A single int (e.g. "1" for pixel-wise, "4" for 4x4 region routing) or a comma-separated per-stage list (e.g. "1,2,4,8" for hierarchical layer-wise routing).',
        default="1",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Number of experts activated per (region) location. top_k=1 yields hard Switch-style routing (purely sparse, no fusion); top_k>=2 enables soft top-k weighted combination on top of the sparse selection.",
        default=2,
    )
    parser.add_argument(
        "--aux_loss_weight",
        type=float,
        help="Coefficient for the Switch-style load-balancing auxiliary loss. Set 0.0 to disable.",
        default=0.01,
    )
    parser.add_argument(
        "--input_routing",
        type=str,
        choices=["full", "sparse"],
        help='How each expert sees the input feature map. "full" (V-MoE): expert always receives the full map; sparsity is only on the gated *output*. "sparse" (legacy): zero-out unrouted pixels before each expert (original behaviour).',
        default="sparse",
    )
    parser.add_argument(
        "--save_routing",
        type=str,
        choices=["none", "lite", "full"],
        help='Whether to dump per-expert routing maps to disk. "none": never save -- fastest training. "lite": save only entropy + per-expert load stats. "full": also save indices / gating maps + a batch of input/labels (large, only enable when needed for heatmap visualization).',
        default="lite",
    )
    parser.add_argument(
        "--routing_epochs",
        type=str,
        help="Comma-separated epochs at which to dump routing snapshots when --save_routing != none. Set to an empty string to disable per-epoch saving.",
        default="0,1,5,10,50,100,200,300,500,700,999",
    )
    parser.add_argument(
        "--save_intermediate_ckpt",
        action="store_true",
        help="Also save a model checkpoint at every --routing_epochs entry (off by default; the best-IoU checkpoint is always saved separately).",
    )
    parser.add_argument("--save_dir", default=None, help="运行名，保存在 result/ 下")
    return parser


def parse_args(argv=None, *, initialize_run=True):
    """解析并规范化参数；测试/工具可用 initialize_run=False 禁止创建目录。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.moe_stages = (
            [bool(x) for x in _integers(args.moe_stages)] if args.moe_stages else None
        )
        args.dilations = _integers(args.dilations)
        sizes = _integers(args.patch_size) if args.patch_size else [1]
        args.patch_size = sizes[0] if len(sizes) == 1 else sizes
        args.routing_epochs = set(_integers(args.routing_epochs or ""))
        if args.moe_stages is not None and len(args.moe_stages) != 4:
            raise ValueError("moe_stages requires four resolution-level entries")
        if len(args.dilations) != 4 or min(args.dilations) < 1:
            raise ValueError("dilations requires four positive integers")
        if len(sizes) not in (1, 4) or min(sizes) < 1:
            raise ValueError("patch_size requires one or four positive integers")
        if args.top_k < 1 or args.top_k > 4:
            raise ValueError("top_k must be between 1 and 4")
    except ValueError as exc:
        parser.error(str(exc))
    if initialize_run:
        # 延迟导入数据工具，让 --help 和参数检查无需加载图像处理/GPU 依赖。
        from pathlib import Path
        from model.utils import make_dir, save_train_log

        if args.save_dir:
            name = Path(args.save_dir)
            if name.is_absolute() or len(name.parts) != 1 or str(name) in {".", ".."}:
                parser.error("save_dir must be a run name under result/, not a path")
            run = Path("result") / name
            run.mkdir(parents=True, exist_ok=False)
        else:
            args.save_dir = make_dir(args.deep_supervision, args.dataset, args.model)
        save_train_log(args, args.save_dir)
    return args
