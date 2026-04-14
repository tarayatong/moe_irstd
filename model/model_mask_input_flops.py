import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mix_blocks import *
from model.mask import *
from model.mobilenet_v2_dg_util import *
import logging

logging.getLogger('thop').setLevel(logging.WARNING)
def model_info(model, input_size):
    input = torch.rand(input_size).cuda()
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    # print('thop| gflops:%.2fG  params:%.2fM'%(flops/ 1E9, params/ 1e6))
    # print(f"model info| summary: {len(list(model.modules()))} layers, {n_p /1E6}M parameters, {n_g /1E6}M gradients{flops/1E9}")
    return n_p, flops

class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,stage=4, block_count=4):   # [16, 32, 64, 128, 256] [2,2,2,2]
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.stage = stage
        self.block = block_count
        self.node_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) for _ in range(block_count)])
        self.channel_squeeze_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) for _ in range(block_count)])
        self.up_list = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True) for i in range(stage)])
        self.nb_filter = nb_filter
        # building first layer
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])

        for j in range (self.block):
            for i in range(self.stage):
                # inp_c = nb_filter[0]*((j-1)*(2**self.stage-1)+(2**(i+1)-1)) # all fore
                if i+j==0:
                    continue
                inp_c = nb_filter[i]*(j)+nb_filter[0]*(2**(i)-1)  # cross
                self.channel_squeeze_list[i][j] = Mask_c(inp_c, inp_c)
                self.node_list[i][j] = self._make_layer(block, inp_c, nb_filter[i], stride=1)

        # for i, blk in enumerate(block_list):
        #     flops_list[i] = model_info(blk, ())

        # img_channels = 3      # ??3??
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(img_channels, nb_filter[0], 3, 1, 1, bias=False),    # s=2 24*40*40
        #     nn.BatchNorm2d(nb_filter[0]),
        #     nn.ReLU(inplace=True),
        # )
        # self.arch_block = nn.Parameter(torch.ones(15, 4))
        self.conv1x1 = nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) 
        for i in range(self.stage):
            self.conv1x1[i] = nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_4_final = self._make_layer(block, nb_filter[0]*self.stage, nb_filter[0], 1)


        self.final_list = nn.ModuleList([nn.ModuleList([]) for _ in range(block_count)])
        for i in range(self.block):
            self.final_list[i] = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)


    def _make_layer(self, block, input_channels, output_channels, num_blocks=2, stride=1):
        layers = []
        layers.append(block(input_channels, output_channels, stride=stride))
        for i in range(num_blocks-1):
            layers.extend([block(output_channels, output_channels, stride=1)])
        return nn.Sequential(*layers)

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if 'arch' in name]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'arch' not in name]
    
    def weight_concat(self, feats, weight):
        '''
        feat: bcwh
        feats: feat list
        weight: 
        '''
        # featchn = [feat.shape[1] for feat in feats]
        # weightsplit = torch.split(weight, split_size_or_sections=featchn, dim=1)
        w = torch.sigmoid(weight)
        new = torch.concat([feat*w[i] for i,feat in enumerate(feats)], dim=1)
        return new


    def forward(self, input):   #, label, den_target, lbda, gamma, p
        batch_num, _, _, _ = input.shape
        # norm_2 = torch.zeros(1, batch_num+1).to(input.device)
        # flops = torch.zeros(1, batch_num+2).to(input.device)
        feat_list = [[[] for _ in range(self.stage)] for _ in range(self.block)]
        # self.arch_block = torch.softmax(self.arch_block, dim=-1)
        feat_list[0][0] = self.conv0_0(input)  #  in (b 3 256 256)  x0_0 (b 16 256 256)
        _, flops_0 = model_info(self.conv0_0, input_size=input.shape)
        flops_full = torch.tensor([flops_0])
        flops_masked = torch.tensor([flops_0])
        # norm_2 = torch.tensor([3.]).to(input.device)
        def cat_input(feat_list, stage, block):
            if block > 0:
                b, c, w, h = feat_list[stage][block-1].size()
            else:
                b, c, w, h = feat_list[stage-1][block].size() 
                w = w//2
                h = h//2
            in_j = torch.cat([feat_list[stage][j] for j in range(block)], 1) if block > 0 else None
            in_i = torch.cat([F.interpolate(feat_list[i][block], (w,h), mode='bilinear') for i in range(stage)], 1) if stage > 0 else None
            if in_i is None and in_j is None:
                return None
            elif in_i is None:
                return in_j
            elif in_j is None:
                return in_i
            else:
                return torch.cat([in_j, in_i], dim=1)
            # return torch.cat([in_j, in_i], 1)  if in_j is not None and in_i is not None else in_j or in_i


        for j in range(self.block):
            for i in range(self.stage): 
                if i+j==0:
                    continue
                in_feat = cat_input(feat_list, i, j)
                mask_c, norm_c, norm_c_t = self.channel_squeeze_list[i][j](in_feat)
                feat_list[i][j] = self.node_list[i][j](in_feat*mask_c) 
                _,flops_blk_full = model_info(self.node_list[i][j], (batch_num, norm_c_t, in_feat.shape[-2], in_feat.shape[-1]))
                flops_blk_masked = flops_blk_full-(norm_c_t-norm_c)*9*self.nb_filter[i]*in_feat.shape[-2]*in_feat.shape[-1]
                flops_masked = torch.cat((flops_masked, torch.tensor([flops_blk_masked])))
                flops_full = torch.cat((flops_full, torch.tensor([flops_blk_full])))

        Final_ = self.conv0_4_final(
            torch.concat([self.up_list[i](self.conv1x1[i](feat_list[i][self.block-1])) if i>0 else feat_list[i][self.block-1] for i in range(self.stage)], 1)) # Final_x0_4 (b 16 256 256)
        
        output = [[] for _ in range(self.stage)]
        for i in range(self.block-1):
            output[i] = self.final_list[i](feat_list[0][i])
        output[self.stage-1] = self.final_list[self.stage-1](Final_)
        flops_real = torch.tensor(flops_masked).sum()
        flops_org = torch.tensor(flops_full).sum()
        return output, flops_real, flops_org# , maskconv

        




