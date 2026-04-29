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
                        help='dataset name: NUDT-SIRST, NUAA-SIRST, NUST-SIRST')
    parser.add_argument('--st_model', type=str, default='',
                        help='NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS,NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS,'
                             'NUAA-SIRST_DNANet_28_07_2021_05_21_33_wDS')
    parser.add_argument('--model_dir', type=str,
                        default = 'NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar',
                        help    = 'NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar,'
                                  'NUAA-SIRST_DNANet_28_07_2021_05_21_33_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
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
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')
    # new
    parser.add_argument('--loss', type=str, default='SoftIoULoss',
                        help="Loss Function. Choose from 'SoftIoU', 'weighted_l2' or 'combined_loss'")
    parser.add_argument('--label', type=str, default='binary',
                        help="Label Type. Choose From 'binary' or 'gaussian'.")
    parser.add_argument('--block', type=str, default='Res_CBAM',
                        help='Res_CBAM or Shuffle')
    parser.add_argument('--moe_stages', type=str, default='1,1,1,1',
                        help='which stages use MoE, e.g. 1,1,1,1 for all, 0,0,1,1 for low-res only')
    parser.add_argument('--dilations', type=str, default='1,2,2,3',
                        help='dilation rates for BasicRFB_a branches, e.g. 1,2,2,3')
    # 必须与训练 build 模型时一致，否则 load_state_dict 虽能载入权重，
    # MoE 前向路由粒度/router 噪声等与训练不同，指标会明显偏差。
    parser.add_argument('--noise_scale', type=float, default=0.2,
                        help='SpatialNoisyTopkRouter alpha; 必须与训练 ./train_spar 一致')
    parser.add_argument('--patch_size', type=str, default='1',
                        help='MoE router 粒度：单整数或逗号分隔 per-stage 列表，须与训练一致')

    args = parser.parse_args()
    args.moe_stages = [bool(int(x)) for x in args.moe_stages.split(',')]
    args.dilations = [int(x) for x in args.dilations.split(',')]
    if args.patch_size is None or args.patch_size == '':
        args.patch_size = 1
    elif ',' in args.patch_size:
        args.patch_size = [int(x) for x in args.patch_size.split(',') if x.strip() != '']
    else:
        args.patch_size = int(args.patch_size)

    # the parser
    return args