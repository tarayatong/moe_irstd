import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
import math
# from model.param_count import DNANet_prune_count
from model.mix_blocks import Res_CBAM_block
import cv2

def SoftIoULoss( pred, target):
        # Old One
        pred = torch.sigmoid(pred)  # B 1 256 256
        smooth = 1
        #miou/iou
        intersection = pred * target    # B 1 256 256
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)
        #niou
        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)
        loss = 1 - loss.mean()  # 什么维度
        # loss = (1 - loss).mean()
        return loss

def bce_loss(pred, target, pos_weight=None):
    """
    简化版二元交叉熵损失
    
    Args:
        pred (Tensor): 预测值 [B,1,H,W]
        target (Tensor): 目标值 [B,1,H,W]
        pos_weight (float): 正样本权重
    """
    # blurred_targets = []
    # weight_maps = []
    # for t in target:
    #     target_np = t.squeeze().cpu().numpy()
    #     from scipy.ndimage import gaussian_filter
    #     blurred = gaussian_filter(target_np, sigma=0.5)
    #     blurred_t = torch.tensor(blurred).unsqueeze(0).unsqueeze(0)
    #     blurred_targets.append(blurred_t)

    #     edges = cv2.Canny((target_np * 255).astype(np.uint8), 50, 150) / 255.0
    #     weight_map = edges + 1  # 在原始基础上增加边缘权重
    #     weight_map = torch.tensor(weight_map).unsqueeze(0).unsqueeze(0)
    #     weight_maps.append(weight_map)
    # blurred_targets = torch.cat(blurred_targets, dim=0)
    # weight_maps = torch.cat(weight_maps, dim=0)

    # if weight_maps is not None:
    #     criterion = nn.BCEWithLogitsLoss(weight=weight_maps.cuda())
    # else:
    criterion = nn.BCEWithLogitsLoss()
    
    return criterion(pred, target)  #blurred_targets.cuda()
    
def spar_iou_loss(output, targets):
    closs= 0
    for i, pred in enumerate(output):
        closs += SoftIoULoss(pred, targets) + bce_loss(pred, targets)
    closs /= len(output)
    # closs = SoftIoULoss(output, targets) + bce_loss(output, targets)
    return closs

class spar_loss(nn.Module):
    def __init__(self):
        super(spar_loss, self).__init__()

    def forward(self, flops_real, flops_ori, den_target, lbda):
        # total sparsity
        # flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        # # block flops
        # flops_conv = flops_tensor[0:batch_size,:].mean(0).sum()
        # flops_mask = flops_mask.mean(0).sum()
        # flops_ori = flops_ori.mean(0).sum() + flops_conv1.mean() + flops_fc.mean()
        # flops_real = flops_conv + flops_mask + flops_conv1.mean() + flops_fc.mean()
        # loss
        rloss = lbda * (flops_real / flops_ori - den_target)**2
        return rloss


class blance_loss(nn.Module):
    def __init__(self):
        super(blance_loss, self).__init__()

    def forward(self, mask_norm_s, mask_norm_c, norm_s_t, norm_c_t, batch_size, 
                den_target, gamma, p):
        norm_s = mask_norm_s
        norm_s_t = norm_s_t.mean(0)
        norm_c = mask_norm_c
        norm_c_t = norm_c_t.mean(0)
        den_s = norm_s[0:batch_size,:].mean(0) / norm_s_t
        den_c = norm_c[0:batch_size,:].mean(0) / norm_c_t
        den_tar = math.sqrt(den_target)
        bloss_s = get_bloss_basic(den_s, den_tar, batch_size, gamma, p)
        bloss_c = get_bloss_basic(den_c, den_tar, batch_size, gamma, p)
        bloss = bloss_s + bloss_c
        return bloss


def get_bloss_basic(spar, spar_tar, batch_size, gamma, p):
    # bound
    bloss_l = (F.relu(p*spar_tar-spar)**2).mean()
    bloss_u = (F.relu(spar-1+p-p*spar_tar)**2).mean()
    bloss = gamma * (bloss_l + bloss_u)
    return bloss 

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        # self.task_loss = nn.CrossEntropyLoss()
        self.spar_loss = spar_loss()
        # self.balance_loss = blance_loss()
    
    def forward(self, output, targets, flops_real, flops_ori, 
                den_target, lbda):
        closs = SoftIoULoss(output, targets)
        sloss = self.spar_loss(flops_real, flops_ori, den_target, lbda)
        # bloss = self.balance_loss(mask_norm_s, mask_norm_c, norm_s_t, norm_c_t, batch_size,
        #                           den_target, gamma, p)
        return closs, sloss #, bloss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


