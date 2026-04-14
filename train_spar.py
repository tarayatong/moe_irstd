# torch and visulization
import time
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
# from model.datasets import PointAnnoLoader, TestPointAnnoLoader
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param, load_dataset_5folders

# model
from model.model_DNANet import  Res_CBAM_block, ShuffleV2Block
# from model.model_TriaNetv3_dilatedblock import ShuffleBlock
# from model.model_DNANet_RFB import BasicRFB_a
from model.model_DNANet import  DNANet
# from model.model_UNet import UNet
# from model.model_UNet_s2 import UNets2
# from model.model_TriaNetv2_dilatedbranch import TriaNetv2
# from model.model_TriaNetv3_dilatedblock import TriaNetv3
# from model.model_TriaNet_RFBv3 import TriaNet_RFBv3
from model.mobilenet_v2_dg_util import InvertedResidual
from model.misc import *
import logging
from model.mask import SpatialSparseMoE

# wandb.init(project='my-awsome-project')
# config = wandb.config
#random seed
class Trainer(object):
    def __init__(self, args):
        seed=40
        random.seed(seed)                          # Python 内建随机模块
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PR = P_R_F1(1, 10, args.crop_size)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_img_ids = load_dataset(args.root, args.dataset, args.split_method)
        if args.mode == 'SIATD10seq':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_img_ids = load_dataset_5folders(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,suffix=args.suffix, dataset_name=args.dataset)
        testset         = TestSetLoader (dataset_dir,img_id=test_img_ids,base_size=args.base_size, crop_size=args.crop_size,suffix=args.suffix, dataset_name=args.dataset)
        # trainset        = PointAnnoLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix, dataset_name=args.dataset)
        # testset         = TestPointAnnoLoader (dataset_dir,img_id=test_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix, dataset_name=args.dataset)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            from model.model_mask_s_shape import DNANet
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, moe_stages=args.moe_stages)   # , batch_size=args.train_batch_size
        elif args.model   == 's4decode':
            from model.model_decode import DNANet
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter)
        elif args.model   == 'MambaIR':
            from model.mambairv2light import MambaIRv2Light
            model = MambaIRv2Light(
                        batch_size=args.train_batch_size,
                        upscale=1,
                        img_size=256,
                        embed_dim=48,
                        d_state=8,
                        depths=[2, 3, 3, 2],
                        num_heads=[4, 4, 4, 4],
                        window_size=16,
                        inner_rank=32,
                        num_tokens=64,
                        convffn_kernel_size=5,
                        img_range=1.,
                        mlp_ratio=1.,
                        upsampler='pixelshuffledirect')
        
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model
        # print(model)

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            # self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()

        self.lossfunc = spar_iou_loss
        # Evaluation metrics
        self.best_iou       = 0
        self.best_f1       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

    def model_info(self, model, input):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in model.parameters())  # number parameters
        n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
        from thop import profile
        flops, params = profile(model, inputs=(input, ))
        # model.forward = original_forward
        fs = f', {flops / 1E9} GFLOPs'  # 640x640 GFLOPs
        # print('thop| gflops:%.2fG  params:%.2fM'%(flops/ 1E9, params/ 1e6))
        print(f"model info| summary: {len(list(model.modules()))} layers, {n_p /1E6}M parameters, {n_g /1E6}M gradients{fs}")

    # Training
    def training(self,epoch):   # 数据 模型 损失   
        tbar = tqdm(self.train_data)    # 终端显示进度条 tqdm参数是dataloader
        self.model.train()  # 初始化定义模型为DNANet，放在cuda上
        losses = AverageMeter()     # 损失类 初始为0

        # 使用torch.cuda.profiler.profile函数包装训练代码
        for i, (data, labels) in enumerate(tbar):  # dataset getitem的返回形式
            data   = data.cuda()    # 放到cuda normed
            labels = labels.cuda()  # torch.Size([16, 1, 256, 256]) max1

            torch.cuda.synchronize()
            start = time.time()
            preds = self.model(data)
            loss = self.lossfunc(preds, labels)
            pred =preds[-1] # 最终结果是最后的输出图
            self.optimizer.zero_grad()  # 优化器初始化
            loss.backward() # 损失回传
            self.optimizer.step()   # 优化迭代
            losses.update(loss.item(), pred.size(0))    # AverageMeter这个类中写更新方法 损失项求均值
              # 终端输出进度
            torch.cuda.synchronize()
            end = time.time()
            infer_time = end-start
            tbar.set_description('Epoch %d, training loss %.4f, iou loss: %.4f, FPS %.4f' % (epoch, losses.avg, loss.mean(), args.train_batch_size/infer_time))
        self.train_loss = losses.avg    # 最终损失是平均值

    def save_router_params(self, model, epoch, save_dir='./router_params'):
        import os
        import numpy as np
        from model.mobile_mamba import MobileMambaBlock
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'router_epoch_{epoch}.txt')
        with open(save_path, 'w') as f:
            # 获取router参数
            for i in range(len(model.node_list)):
                for j in range(len(model.node_list[i])):
                    for k, module in enumerate(model.node_list[i][j]):
                        if isinstance(module, MobileMambaBlock):
                            router_params = module.mixer.m.attn.spacial_moe.router.detach().cpu().numpy()
                            # 获取参数形状
                            param_shape = router_params.shape
                            # 写入模块信息和参数形状
                            f.write(f"Node {i}, Block {j}, Module {k} - Shape: {param_shape}\n")
                            # 写入参数值，格式化为矩阵形式
                            for row in router_params:
                                param_str = ' '.join(f"{val:.4f}" for val in row)
                                f.write(f"  {param_str}\n")
                            # 添加空行分隔不同
                            f.write("\n")

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.PR.reset()
        losses = AverageMeter()
        all_infer_time = 0
        with torch.no_grad():   # 梯度不更新
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                torch.cuda.synchronize()
                start = time.time()
                preds = self.model(data)
                loss = self.lossfunc(preds, labels)
                pred =preds[-1] # 最终结果是最后的输出图
                torch.cuda.synchronize()
                end = time.time()
                infer_time = end-start
                all_infer_time += infer_time
                losses.update(loss.item(), pred.size(0))
                # miou & ROC
                self.ROC .update(pred, labels)  # 根据结果计算ROC曲线
                self.mIoU.update(pred, labels)  # 计算mIoU
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU, _ = self.mIoU.get()   
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))
                # P R F1
                # self.PR .update(pred, labels)
                # P, R, F1 = self.PR.get()
                # tbar.set_description('Epoch %d, test loss %.4f, F1: %.4f, P: %.4f, R: %.4f' % (epoch, losses.avg, F1.max(), P[F1.argmax()], R[F1.argmax()] )) 
            test_loss=losses.avg
            print('FPS: %.2f' % (len(self.test_data)*args.test_batch_size/all_infer_time))
        # save high-performance model
        save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
        if mean_IOU > self.best_iou:
            self.best_iou = mean_IOU 

        # save_model_F1(F1, self.best_f1, self.save_dir, self.save_prefix,
        #            self.train_loss, test_loss, R, P, epoch, self.model.state_dict())
        # if F1.max() > self.best_f1:
        #     self.best_f1 = F1.max()

def main(args):
    # torch.cuda.manual_seed(1000)
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)
    input = torch.randn(1, 3, 256, 256).cuda()
    trainer.model_info(trainer.model, input)
    
        # if epoch in [0,1,5,10,50,100,200,300,400,499]:
        #     trainer.save_router_params(trainer.model, epoch)


if __name__ == "__main__":

    # logging.getLogger('thop').setLevel(logging.WARNING)
    args = parse_args()
    main(args)





