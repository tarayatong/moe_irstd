import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mix_blocks import *
from model.mask import *
from model.mobilenet_v2_dg_util import *
from prettytable import PrettyTable
import logging

class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,stage=2, block_count=2,width_mult=1.0, inverted_residual_setting=None, 
                 round_nearest=8, in_size=(224, 224), loss=None, **kwargs):   # [16, 32, 64, 128, 256] [2,2,2,2]
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.stage = stage
        self.block = block_count
        self.criterion = loss
        self.node_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) for _ in range(block_count)])
        self.channel_squeeze_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) for _ in range(block_count)])
        self.up_list = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True) for i in range(stage)])
        # input_channel = 32
        # last_channel = 1280
        h, w = in_size

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, tile
                [1, 16, 1, 1, 16],
                [6, 24, 2, 2, 8],
                [6, 32, 3, 2, 4],
                [6, 64, 4, 2, 2],
                [6, 96, 3, 1, 2],
                [6, 160, 3, 2, 2],
                [6, 320, 1, 1, 2],
            ]
        expand_ratio = [6 if i>0 else 1 for i in range(block_count)]
        size_list = [int(h/(2**i)) for i in range(self.stage)]
        tile_list = [16/(2**i) if i<4 else 2 for i in range(block_count)]

        # building first layer
        self.conv0_0 = ConvBNReLU_1st(3, nb_filter[0], stride=1)
        self.flops_conv1 = torch.Tensor([3 * h//2 * w//2 * 3 * nb_filter[0]])
        for j in range (self.block):
            for i in range(self.stage):
                # inp_c = nb_filter[0]*((j-1)*(2**self.stage-1)+(2**(i+1)-1)) # all fore
                if i+j==0:
                    continue
                inp_c = nb_filter[i]*(j)+nb_filter[0]*(2**(i)-1)  # cross
                # mask_conv = self._make_layer(block, inp_c, nb_filter[i], stride=1)
                self.node_list[i][j] = block(inp_c,  nb_filter[i], stride=1, 
                                      expand_ratio=expand_ratio[j], h=size_list[i], w=size_list[i], eta=tile_list[i], **kwargs)
                # self.channel_squeeze_list[i][j] = Mask_c(inp_c, nb_filter[i])


        # img_channels = 3      # ??3??
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(img_channels, nb_filter[0], 3, 1, 1, bias=False),    # s=2 24*40*40
        #     nn.BatchNorm2d(nb_filter[0]),
        #     nn.ReLU(inplace=True),
        # )
        self.arch_block = nn.Parameter(torch.ones(15, 4))
        self.conv1x1 = nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) 
        for i in range(self.stage):
            self.conv1x1[i] = nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_4_final = ConvBNReLU_1st(nb_filter[0]*self.stage, nb_filter[0], 1)
        self.flops_fc = torch.Tensor([nb_filter[0]*self.stage * nb_filter[0] *h*w])

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


    def forward(self, input, label, den_target, lbda, gamma, p):
        batch_size, _, _, _ = input.size()
        norm1 = torch.zeros(1, batch_size+1).to(input.device)
        norm2 = torch.zeros(1, batch_size+1).to(input.device)
        flops = torch.zeros(1, batch_size+2).to(input.device)
        feat_list = [[[] for _ in range(self.stage)] for _ in range(self.block)]
        # self.arch_block = torch.softmax(self.arch_block, dim=-1)
        x0, norm1, norm2, flops = self.conv0_0((input, norm1, norm2, flops))  #  in (b 3 256 256)  x0_0 (b 16 256 256)
        feat_list[0][0] =x0
        
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
                # x, norm1, norm2, flops = self.channel_squeeze_list[i][j](in_feat)
                x, norm1, norm2, flops= self.node_list[i][j]((in_feat, norm1, norm2, flops)) 
                feat_list[i][j] = x

        Final_, norm1, norm2, flops = self.conv0_4_final((
            torch.concat([self.up_list[i](self.conv1x1[i](feat_list[i][self.block-1])) if i>0 else feat_list[i][self.block-1] for i in range(self.stage)], 1), norm1, norm2, flops)) # Final_x0_4 (b 16 256 256)
        
            # 

        output = [[] for _ in range(self.stage)]
        for i in range(self.stage-1):
            output[i] = self.final_list[i](feat_list[i][self.block-1])
        output[self.stage-1] = self.final_list[self.stage-1](Final_)
        # return output # , maskconv


        # norm and flops
        norm_s = norm1[1:, 0:batch_size].permute(1, 0).contiguous()
        norm_c = norm2[1:, 0:batch_size].permute(1, 0).contiguous()
        norm_s_t = norm1[1:, -1].unsqueeze(0)
        norm_c_t = norm2[1:, -1].unsqueeze(0)
        flops_real = [flops[1:, 0:batch_size].permute(1, 0).contiguous(), 
                      self.flops_conv1.to(x.device), self.flops_fc.to(x.device)]
        flops_mask = flops[1:, -2].unsqueeze(0)
        flops_ori  = flops[1:, -1].unsqueeze(0)
        # get outputs
        outputs = {}
        outputs["closs"], outputs["rloss"], outputs["bloss"] = self.get_loss(
                            output[self.stage-1], label, batch_size, den_target, lbda, gamma, p,
                            norm_s, norm_c, norm_s_t, norm_c_t, 
                            flops_real, flops_mask, flops_ori)
        outputs["out"] = output[self.stage-1]
        outputs["flops_real"] = flops_real
        outputs["flops_mask"] = flops_mask
        outputs["flops_ori"] = flops_ori
        return outputs
            # return output   # , maskconv
    def set_criterion(self, criterion):
        self.criterion = criterion
        return
    
    def get_loss(self, output, label, batch_size, den_target, lbda, gamma, p,
                 mask_norm_s, mask_norm_c, norm_s_t, norm_c_t,
                 flops_real, flops_mask, flops_ori):
        closs, rloss, bloss = self.criterion(output, label, flops_real, flops_mask,
                flops_ori, batch_size, den_target, lbda, mask_norm_s, mask_norm_c,
                norm_s_t, norm_c_t, gamma, p)
        return closs, rloss, bloss
    
    def record_flops(self, flops_conv, flops_mask, flops_ori, flops_conv1, flops_fc):
        i = 0
        table = PrettyTable(['Layer', 'Conv FLOPs', 'Conv %', 'Mask FLOPs', 'Total FLOPs', 'Total %', 'Original FLOPs'])
        table.add_row(['layer0'] + ['{flops:.2f}K'.format(flops=flops_conv1/1024)] + [' ' for _ in range(5)])
        for name, m in self.named_modules():
            if isinstance(m, InvertedResidual):
                table.add_row([name] + ['{flops:.2f}K'.format(flops=flops_conv[i]/1024)] + ['{per_f:.2f}%'.format( 
                    per_f=flops_conv[i]/flops_ori[i]*100)] + ['{mask:.2f}K'.format(mask=flops_mask[i]/1024)] +
                    ['{total:.2f}K'.format(total=(flops_conv[i]+flops_mask[i])/1024)] + ['{per_t:.2f}%'.format(
                    per_t=(flops_conv[i]+flops_mask[i])/flops_ori[i]*100)] +
                    ['{ori:.2f}K'.format(ori=flops_ori[i]/1024)])
                i+=1
        table.add_row(['fc'] + ['{flops:.2f}K'.format(flops=flops_fc/1024)] + [' ' for _ in range(5)])
        table.add_row(['Total'] + ['{flops:.2f}K'.format(flops=(flops_conv[i]+flops_conv1+flops_fc)/1024)] + 
                    ['{per_f:.2f}%'.format(per_f=(flops_conv[i]+flops_conv1+flops_fc)/(flops_ori[i]+flops_conv1+flops_fc)*100)] + 
                    ['{mask:.2f}K'.format(mask=flops_mask[i]/1024)] + ['{total:.2f}K'.format(
                    total=(flops_conv[i]+flops_mask[i]+flops_conv1+flops_fc)/1024)] + ['{per_t:.2f}%'.format(
                    per_t=(flops_conv[i]+flops_mask[i]+flops_conv1+flops_fc)/(flops_ori[i]+flops_conv1+flops_fc)*100)] +
                    ['{ori:.2f}K'.format(ori=(flops_ori[i]+flops_conv1+flops_fc)/1024)])
        logging.info('\n{}'.format(table))

        




