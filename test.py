# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio
import time
# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
# from model.datasets import PointAnnoLoader, TestPointAnnoLoader
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param, load_dataset_5folders

# Model
from model.model_mask_s_shape import  DNANet
from model.model_DNANet import Res_CBAM_block

class Trainer(object):
    def __init__(self, args):
        args.base_size = 256
        args.crop_size = 256
        args.st_model = 'NUAA-SIRST_DNANet_28_05_2025_15_46_52_wDS'
        args.model_dir = 'NUAA-SIRST_DNANet_28_05_2025_15_46_52_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar'
        args.model = 'DNANet'
        args.dataset = 'NUAA-SIRST'
        args.split_method = '50_50'
        args.backbone = 'resnet_18'
        args.test_batch_size = 1
        args.mode = 'TXT'

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr, args.crop_size)
        self.PR = P_R_F1(1,args.ROC_thr, args.crop_size)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        if args.deep_supervision=='True' :
            deep_supervision = True 
        elif args.deep_supervision=='False':
            deep_supervision = False 
        result_dir = './result/'

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, _, val_img_ids=load_dataset(args.root, args.dataset,args.split_method)
        if args.mode == 'SIATD10seq':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, _, val_img_ids = load_dataset_5folders(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix, dataset_name=args.dataset)
        # testset         = TestPointAnnoLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix, dataset_name=args.dataset)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)

        model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter)

        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model
        self.lossfunc = spar_iou_loss

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = torch.load(result_dir + args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        all_infer_time = 0
        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                # torch.cuda.synchronize()
                # list = [19, 236, 238, 297, 300, 307, 312, 329, 332, 406, 514, 531, 539, 577, 579, 642]
                if True:
                    # print(i)
                    start = time.time()
                    preds = self.model(data)
                    loss = self.lossfunc(preds, labels)
                    pred =preds[-1] # 最终结果是最后的输出图
                    torch.cuda.synchronize()
                    end = time.time()
                    infer_time = end-start
                    all_infer_time += infer_time
                    # list =  [0, 6, 8, 18, 22, 24, 32, 33, 38, 44, 53, 87, 97, 112, 121, 123, 127, 128, 133, 142, 151, 158, 166, 179, 184, 191, 196] # nuaa [45, 48, 51, 65]  # nudt [19, 236, 238, 297, 300, 307, 312, 329, 332, 406, 514, 531, 539, 577, 579, 642] irstd 
                    # if i in list:
                    #     import torchvision.utils as vutils
                    #     vutils.save_image((pred.sigmoid() > 0.1).float(),
                    #                     './vis_result/nuaa/%d_moe.png' % i)         

                    losses.    update(loss.item(), pred.size(0))
                    self.ROC.  update(pred, labels)
                    self.mIoU. update(pred, labels)
                    self.PD_FA.update(pred, labels)
                    self.PR.update(pred, labels)
                    # if self.mIoU.iou < 0.5:
                    #     print(i)
                    #     preds = self.model(data)

                    ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                    _, mean_IOU,tiou = self.mIoU.get()
                    tbar.set_description('test loss %.4f, mean_IoU: %.4f' % (losses.avg, mean_IOU ))
            FA, PD = self.PD_FA.get(len(val_img_ids))
            P, R, F1 = self.PR.get()
            FPS = (len(self.test_data) / all_infer_time)
            infer_avg = (all_infer_time / len(self.test_data))
            if os.path.isdir(result_dir + '/' +args.st_model):
                file_name = result_dir + '/' +args.st_model  +'/' +'_mertics_' + '.txt'
            else:
                file_name = result_dir + '/test_result/' + args.dataset + args.st_model +'_metrics_' + '.txt'
            input = torch.randn(1, 3, 256, 256).cuda()
            modelinfo = self.model_info(model, input)
            save_metrics(file_name, args.epochs, P, R, F1, PD, FA, mean_IOU, recall, precision, FPS, infer_avg, modelinfostr=modelinfo, niou=tiou/len(self.test_data))
            # try:
            #     # save_pd_fa(result_dir + '/' +args.st_model  +'/' +'_PD_FA_' + str(255)+'.txt',PD,FA)
            #     save_prf1(result_dir + '/' +args.st_model  +'/' +'_PRF1_' + str(255)+'.txt',args.epochs,P,R,F1)
            # except:
            #     # save_pd_fa(result_dir + '/test_result/' + args.dataset + args.st_model +'_PD_FA_' + str(255)+'.txt',PD,FA)
            #     save_prf1(result_dir + '/test_result/' + args.dataset + args.st_model +'_PRF1_' + str(255)+'.txt',args.epochs,P,R,F1)

            # try:
            #     scio.savemat(result_dir + '/' +args.st_model  +'/' +'_PD_FA_' + str(255),
            #              {'number_record1': FA, 'number_record2': PD})
            # except:
            #     scio.savemat(result_dir + '/' +  'test_result'+ '/' + args.dataset +args.st_model  + '_PD_FA_' + str(255),
            #              {'number_record1': FA, 'number_record2': PD})
            # save_result_for_test(dataset_dir, args.st_model,args.epochs, mean_IOU, recall, precision)
    def model_info(self, model, input):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in model.parameters())  # number parameters
        n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
        from thop import profile
        flops, params = profile(model, inputs=(input, ))
        fs = f', {flops / 1E9} GFLOPs'  # 640x640 GFLOPs
        # print('thop| gflops:%.2fG  params:%.2fM'%(flops/ 1E9, params/ 1e6))
        print(f"model info| summary: {len(list(model.modules()))} layers, {n_p /1E6}M parameters, {n_g /1E6}M gradients{fs}")
        return f"model info| summary: {len(list(model.modules()))} layers, {n_p /1E6}M parameters, {n_g /1E6}M gradients{fs}"


def main(args):
    trainer = Trainer(args)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)





