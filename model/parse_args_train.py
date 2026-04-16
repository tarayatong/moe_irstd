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

    args = parser.parse_args()
    args.base_size = 256
    args.crop_size = 256
    args.epochs = 1000
    args.dataset = 'NUAA-SIRST'
    args.split_method = '50_50'
    args.model = 'DNANet'
    args.backbone = 'resnet_18'
    args.train_batch_size = 8
    args.test_batch_size = 8
    args.moe_stages = None
    args.dilations = '1,1,1,1'
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
    # make dir for save result
    args.save_dir = make_dir(args.deep_supervision, args.dataset, args.model)
    # save training log
    save_train_log(args, args.save_dir)
    # the parser
    return args