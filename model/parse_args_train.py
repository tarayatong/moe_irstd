from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='DNANet',
                        help='model name: DNANet')
    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False (model==DNANet)')


    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='dataset name:  NUDT-SIRST, NUAA-SIRST, NUST-SIRST')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='50_50, 10000_100(for NUST-SIRST)')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')
    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    
    # new
    parser.add_argument('--loss', type=str, default=None,
                        help="Loss Function. Choose from 'SoftIoU', 'weighted_l2' or 'combined_loss'")
    parser.add_argument('--label', type=str, default='binary',
                        help="Label Type. Choose From 'binary' or 'gaussian'.")
    parser.add_argument('--block', type=str, default='Res_CBAM',
                        help='Res_CBAM or Shuffle')
    # 添加compile相关参数
    parser.add_argument('--use_compile', type=bool, default=True,
                      help='whether to use torch.compile')
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                      choices=['default', 'reduce-overhead', 'max-autotune'],
                      help='torch.compile optimization mode')
    # archi_para
    parser.add_argument('--arch_lr', type=float, default=0.05,
                    help='learning rate for alpha and beta in architect searching process')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--moe_stages', type=str, default='1,1,1,1',
                        help='which stages use MoE, e.g. 1,1,1,1 for all, 0,0,1,1 for low-res only')
    parser.add_argument('--dilations', type=str, default='1,2,2,3',
                        help='dilation rates for BasicRFB_a branches, e.g. 1,2,2,3')
    parser.add_argument('--noise_scale', type=float, default=0.2,
                        help='noise scale factor alpha for SpatialNoisyTopkRouter')
    parser.add_argument('--patch_size', type=str, default='1',
                        help='MoE routing granularity. A single int (e.g. "1" '
                             'for pixel-wise, "4" for 4x4 region routing) or a '
                             'comma-separated per-stage list (e.g. "1,2,4,8" '
                             'for hierarchical layer-wise routing).')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Number of experts activated per (region) '
                             'location. top_k=1 yields hard Switch-style '
                             'routing (purely sparse, no fusion); top_k>=2 '
                             'enables soft top-k weighted combination on top '
                             'of the sparse selection.')
    parser.add_argument('--aux_loss_weight', type=float, default=1e-2,
                        help='Coefficient for the Switch-style load-balancing '
                             'auxiliary loss. Set 0.0 to disable.')
    parser.add_argument('--input_routing', type=str, default='full',
                        choices=['full', 'sparse'],
                        help='How each expert sees the input feature map. '
                             '"full" (V-MoE): expert always receives the full '
                             'map; sparsity is only on the gated *output*. '
                             '"sparse" (legacy): zero-out unrouted pixels before '
                             'each expert (original behaviour).')

    # ---- routing-map & checkpoint snapshot saving --------------------------
    # Saving routing maps every iteration is expensive, so all snapshots are
    # opt-in via the following flags.
    parser.add_argument('--save_routing', type=str, default='none',
                        choices=['none', 'lite', 'full'],
                        help='Whether to dump per-expert routing maps to disk. '
                             '"none" (default): never save -- fastest training. '
                             '"lite": save only entropy + per-expert load stats. '
                             '"full": also save indices / gating maps + a batch '
                             'of input/labels (large, only enable when needed '
                             'for heatmap visualization).')
    parser.add_argument('--routing_epochs', type=str,
                        default='0,1,5,10,50,100,200,300,500,700,999',
                        help='Comma-separated epochs at which to dump routing '
                             'snapshots when --save_routing != none. Set to '
                             'an empty string to disable per-epoch saving.')
    parser.add_argument('--save_intermediate_ckpt', action='store_true',
                        help='Also save a model checkpoint at every '
                             '--routing_epochs entry (off by default; the '
                             'best-IoU checkpoint is always saved separately).')

    args = parser.parse_args()
    args.base_size = 256
    args.crop_size = 256
    args.epochs = 1000
    args.dataset = 'NUAA-SIRST'
    args.split_method = '50_50'
    args.model = 'DNANet'
    args.backbone = 'resnet_18'
    args.train_batch_size = 4
    args.test_batch_size = 4
    args.moe_stages = '1,1,1,1'
    args.dilations = '1,2,2,3'
    args.noise_scale = 0.2
    # MoE routing granularity (uniform OR per-stage). Examples:
    #   '1'         pixel-wise (paper baseline)
    #   '4'         uniform 4x4 region routing
    #   '1,2,4,8'   layer-wise hierarchical routing
    #   '256'       global per-image routing
    args.patch_size = '1'
    # MoE sparsity & load-balancing.
    #   top_k=1   -> hard Switch routing (single expert per region)
    #   top_k=2   -> soft top-2 weighted combination on top of sparse selection
    # aux_loss_weight is the coefficient on the Switch load-balancing loss;
    # 1e-2 follows Switch Transformer / GShard. Set 0 to ablate.
    args.top_k = 2
    args.aux_loss_weight = 1e-2
    # Expert input: 'full' = V-MoE (all experts see full feature map);
    # 'sparse' = legacy masked input before each expert convolution.
    args.input_routing = 'full'
    # # args.lr = 0.02

    # args.base_size = 512
    # args.crop_size = 512
    # args.epochs = 1000
    # args.dataset = 'IRSTD'
    # args.split_method = '80_20'
    # args.model = 'DNANet'
    # args.backbone = 'resnet_18'
    # args.train_batch_size = 4
    # args.test_batch_size = 4

    args.mode = 'TXT'
    args.loss = 'spar_iou_loss'

    args.moe_stages = [bool(int(x)) for x in args.moe_stages.split(',')] if args.moe_stages else None
    args.dilations = [int(x) for x in args.dilations.split(',')]
    if args.patch_size is None or args.patch_size == '':
        args.patch_size = 1
    elif ',' in args.patch_size:
        args.patch_size = [int(x) for x in args.patch_size.split(',') if x.strip() != '']
    else:
        args.patch_size = int(args.patch_size)

    if args.routing_epochs is None or args.routing_epochs.strip() == '':
        args.routing_epochs = set()
    else:
        args.routing_epochs = {int(x) for x in args.routing_epochs.split(',') if x.strip() != ''}
    # make dir for save result
    args.save_dir = make_dir(args.deep_supervision, args.dataset, args.model)
    # save training log
    save_train_log(args, args.save_dir)
    # the parser
    return args