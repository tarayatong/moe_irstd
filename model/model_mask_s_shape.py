import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mix_blocks import *
from model.blocks import DySample, ACmix, Sobel_tensor
from model.mask import *
from model.mobilenet_v2_dg_util import *
# from model.mambairv2light import Matrix_upsample
import logging
from model.mobile_mamba import MobileMambaBlock

logging.getLogger('thop').setLevel(logging.WARNING)

class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,
                 stage=4, block_count=4, moe_stages=None, dilations=None, noise_scale=0.2,
                 patch_size=1):   # [16, 32, 64, 128, 256] [2,2,2,2]
        """
        Args:
            patch_size: granularity of MoE spatial routing.  Supports
                * ``int``  – the same patch size is used for every stage
                  (router map = feature_map // patch_size).  ``patch_size=1``
                  reproduces the original pixel-wise routing.
                * ``list``/``tuple`` of length ``stage`` – per-stage patch
                  size, e.g. ``[1, 2, 4, 8]`` for hierarchical region routing.
                * ``None`` – treated as ``1`` (pixel-wise).
        """
        super(DNANet, self).__init__()
        if moe_stages is None:
            moe_stages = [True] * stage
        self.moe_stages = moe_stages

        # Normalise ``patch_size`` to a per-stage list so each MoE block can be
        # configured independently (plan B: optional layer-wise ablation).
        if patch_size is None:
            patch_size_list = [1] * stage
        elif isinstance(patch_size, (list, tuple)):
            assert len(patch_size) == stage, (
                f"patch_size list length {len(patch_size)} != stage {stage}")
            patch_size_list = [max(int(p), 1) for p in patch_size]
        else:
            patch_size_list = [max(int(patch_size), 1)] * stage
        self.patch_size_list = patch_size_list
        input_size=512
        self.relu = nn.ReLU(inplace = True)
        self.pool  = nn.MaxPool2d(2, 2)
        # self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.stage = stage
        self.block = block_count
        self.node_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) for _ in range(block_count)])
        self.channel_squeeze_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) for _ in range(block_count)])
        # self.up_list = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True) for i in range(stage)])
        self.up_list = nn.ModuleList([DySample(nb_filter[i], scale=2**i) for i in range(stage)])
        self.nb_filter = nb_filter
        self.connect_type = 'cross'#'cross'
        assert self.connect_type in ['cross', 'fore', 'dense']

        # building first layer
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])

        for j in range (self.block):
            for i in range(self.stage):
                if i+j==0:
                    continue
                if self.connect_type == 'fore':
                    if j!=2:  # 向下
                        inp_c = nb_filter[0]*((j)*(2**self.stage-1)+(2**(i)-1)) # all fore
                    else:   # 向上
                        inp_c = nb_filter[0]*((j)*(2**self.stage-1) + 2**(i+1)*(2**(self.stage-1-i)-1)) 
                elif self.connect_type == 'cross':
                    if j!=2:  # 向下
                        inp_c = nb_filter[i]*(j)+nb_filter[0]*(2**(i)-1)  # cross
                    else:   # 向上
                        inp_c = nb_filter[i]*(j)+nb_filter[0]*2**(i+1)*(2**(self.stage-1-i)-1) 
                elif self.connect_type == 'dense':
                    inp_c = nb_filter[i]*(j)+nb_filter[0]*(2**(i)-1)+(nb_filter[0]*(int(2**(i-1)*7) if i<3 else 12)-nb_filter[i])*min(j, 1)
                        
                    
                # self.channel_squeeze_list[i][j] = Mask_c(inp_c, inp_c)
                # self.channel_squeeze_list[i][j] = ChannelRouter(inp_c, top_k=inp_c//2)
                # self.channel_squeeze_list[i][j] = nn.Conv2d(inp_c, inp_c//2, 1)
                # self.channel_squeeze_list[i][j] = GroupedChannelSelection(inp_c, inp_c//2, 1)
                # self.channel_squeeze_list[i][j] = ChannelAttention(inp_c)
                if self.moe_stages[i]:
                    self.node_list[j][i] = nn.Sequential(
                            ConvBNReLU(inp_c, nb_filter[i], 3),
                            MobileMambaBlock('s', nb_filter[i], 0.7, 0.2, 5, 0, ssm_ratio=2,
                                             layer=i, dilations=dilations,
                                             noise_scale=noise_scale,
                                             patch_size=self.patch_size_list[i]),
                            )
                else:
                    self.node_list[j][i] = self._make_layer(block, inp_c, nb_filter[i], stride=1)
                # if i <2:
                # self.node_list[j][i] = self._make_layer(block, inp_c, nb_filter[i], stride=1)
                # elif i ==2:
                #     self.node_list[j][i] = nn.Sequential(
                #         ConvBNReLU(inp_c, nb_filter[i], 3),
                #         MobileMambaBlock('s', nb_filter[i], 0.8, 0.2, 7, 0, ssm_ratio=2)
                #         )
                # elif i ==3:
                #     self.node_list[j][i] = nn.Sequential(
                #         ConvBNReLU(inp_c, nb_filter[i], 3),
                #         MobileMambaBlock('s', nb_filter[i], 0.7, 0.2, 5, 0, ssm_ratio=2)
                #         )

        # img_channels = 3      # ??3??
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(img_channels, nb_filter[0], 3, 1, 1, bias=False),    # s=2 24*40*40
        #     nn.BatchNorm2d(nb_filter[0]),
        #     nn.ReLU(inplace=True),
        # )
        # self.arch_block = nn.Parameter(torch.ones(15, 4))
        self.conv1x1 = nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) 
        for i in range(self.stage):
            # self.conv1x1[i] = nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=1, stride=1)
            self.conv1x1[i] = nn.Sequential(
                nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU(inplace=True)
            )
        # self.conv0_4_final = self._make_layer(block, nb_filter[0]*self.stage, nb_filter[0], 1)
        self.conv0_4_final = self._make_layer(block, nb_filter[0]*self.stage, nb_filter[0], 1)


        self.final_list = nn.ModuleList([nn.ModuleList([]) for _ in range(block_count)])
        for i in range(self.block):
            self.final_list[i] = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        # self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)


    def _make_layer(self, block, input_channels, output_channels, num_blocks=2, stride=1):
        layers = []
        layers.append(block(input_channels, output_channels, stride=stride))
        for i in range(num_blocks-1):
            layers.extend([block(output_channels, output_channels, stride=1)])
        # if input_channels>=self.nb_filter[0]*(2**self.stage-1):
        #     layers.extend([ACmix(output_channels, output_channels, (4, 4), 8)])
                
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
        batch_num, in_ch, _, _ = input.shape
        # norm_2 = torch.zeros(1, batch_num+1).to(input.device)
        feat_list = [[[] for _ in range(self.stage)] for _ in range(self.block)]
        # self.arch_block = torch.softmax(self.arch_block, dim=-1)
        feat_list[0][0] = self.conv0_0(input)  #  in (b 3 256 256)  x0_0 (b 16 256 256)
        channel_full = torch.tensor([in_ch])
        channel_masked = torch.tensor([in_ch])
        # norm_2 = torch.tensor([3.]).to(input.device)
        def cat_input_cross(feat_list, stage, block):
            if block > 0:
                b, c, w, h = feat_list[block-1][stage].size()
            else:
                b, c, w, h = feat_list[block][stage-1].size() 
                w = w//2
                h = h//2
            in_j = torch.cat([feat_list[j][stage] for j in range(block)], 1) if block > 0 else None
            if block!=2:# down
                in_i = torch.cat([F.interpolate(feat_list[block][i], (w,h), mode='bilinear') for i in range(stage)], 1) if stage > 0 else None
            else:   # up
                # in_i = torch.cat([F.interpolate(feat_list[i][block], (w,h), mode='bilinear') for i in range(self.stage-1, stage, -1)], 1) if stage < self.stage-1 else None
                in_i = torch.cat([F.interpolate(feat_list[block][i], (w,h), mode='bilinear') for i in range(self.stage-1, stage, -1)], 1) if stage < self.stage-1 else None
            if in_i is None and in_j is None:
                return None
            elif in_i is None:
                return in_j
            elif in_j is None:
                return in_i
            else:
                return torch.cat([in_j, in_i], dim=1)
            
        def cat_input_dense(feat_list, stage, block):
            if block > 0:
                b, c, w, h = feat_list[block-1][stage].size()
            else:
                b, c, w, h = feat_list[block][stage-1].size() 
                w = w//2
                h = h//2
            in_j = torch.cat([feat_list[j][stage] for j in range(block-1)], 1) if block > 1 else None
            in_i = torch.cat([F.interpolate(feat_list[block][i], (w,h), mode='bilinear') for i in range(stage)], 1) if stage > 0 else None
            in_dense = torch.cat([F.interpolate(feat_list[block-1][i], (w,h), mode='bilinear') for i in range(max(stage-1, 0), min(self.stage, stage+2))], 1) if block > 0 else None
            valid_tensors = []
            if in_j is not None:
                valid_tensors.append(in_j)
            if in_i is not None:
                valid_tensors.append(in_i)
            if in_dense is not None:
                valid_tensors.append(in_dense)
            if valid_tensors:
                return torch.cat(valid_tensors, 1)  # 在channel维度上拼接
            else:
                return None  # 或者根据需求设置默认值
            
        def cat_input_fore(feat_list, stage, block):
            if block > 0:
                b, c, w, h = feat_list[block-1][stage].size()
            else:
                b, c, w, h = feat_list[block][stage-1].size() 
                w = w//2
                h = h//2
            # in_i = torch.cat([F.interpolate(feat_list[block][i], (w,h), mode='bilinear') for i in range(stage)], 1) if stage > 0 else None
            if block!=2:# down
                in_i = torch.cat([F.interpolate(feat_list[block][i], (w,h), mode='bilinear') for i in range(stage)], 1) if stage > 0 else None
            else:   # up
                in_i = torch.cat([F.interpolate(feat_list[block][i], (w,h), mode='bilinear') for i in range(self.stage-1, stage, -1)], 1) if stage < self.stage-1 else None
            in_fore = torch.cat([F.interpolate(feat_list[j][i], (w,h), mode='bilinear') for i in range(self.stage) for j in range(block)], 1) if block > 0 else None
            if in_i is None and in_fore is None:
                return None
            elif in_i is None:
                return in_fore
            elif in_fore is None:
                return in_i
            else:
                return torch.cat([in_fore, in_i], dim=1)

        for j in range(self.block):
            if j!=2:
                for i in range(self.stage): 
                    if i+j==0:
                        continue
                    if self.connect_type == 'cross':
                        in_feat = cat_input_cross(feat_list, i, j)
                    elif self.connect_type == 'fore':
                        in_feat = cat_input_fore(feat_list, i, j)      
                    elif self.connect_type == 'dense':
                        in_feat = cat_input_dense(feat_list, i, j)        
                    # chn_w = self.channel_squeeze_list[i][j](in_feat)
                    # in_feat = in_feat+in_feat*chn_w
                    # if j==3 and i==0:
                    #     print(j,i)
                    feat_list[j][i] = self.node_list[j][i](in_feat) 
            else:
                for i in range(self.stage-1, -1, -1): 
                    if i+j==0:
                        continue
                    if self.connect_type == 'cross':
                        in_feat = cat_input_cross(feat_list, i, j)
                    elif self.connect_type == 'fore':
                        in_feat = cat_input_fore(feat_list, i, j)      
                    elif self.connect_type == 'dense':
                        in_feat = cat_input_dense(feat_list, i, j)        
                    # chn_w = self.channel_squeeze_list[i][j](in_feat)
                    # in_feat = in_feat+in_feat*chn_w
                    feat_list[j][i] = self.node_list[j][i](in_feat) 

        Final_ = self.conv0_4_final(
            torch.concat([self.conv1x1[i](self.up_list[i](feat_list[self.block-1][i])) if i>0 else feat_list[self.block-1][i] for i in range(self.stage)], 1)) # Final_x0_4 (b 16 256 256)
        
        output = [[] for _ in range(self.block)]
        for i in range(self.block-1):
            output[i] = self.final_list[i](feat_list[i][0])
        output[self.block-1] = self.final_list[self.block-1](Final_)

        # chn_real = torch.tensor(channel_masked).sum()
        # chn_org = torch.tensor(channel_full).sum()
        return output# , maskconv

        




