import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mix_blocks import *
from model.mask import *
from model.mobilenet_v2_dg_util import *
import logging
# from model.qformer_fast import QFormerFast, CrossDecoder
# from model.mamba_moe import ImageMoEMamba
# from model.mambaout import MambaOut
from model.mambairv2light import MambaIRv2Light

logging.getLogger('thop').setLevel(logging.WARNING)


class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,stage=5, block_count=1):   # [16, 32, 64, 128, 256] [2,2,2,2]
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.stage = stage
        self.block = block_count
        self.node_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([]) for _ in range(block_count)]) for _ in range(stage)])
        self.channel_squeeze_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([]) for _ in range(block_count)]) for _ in range(stage)])
        self.up_list = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True) for i in range(stage)])
        self.down_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=2**(stage-i)+1, stride=2**(stage-1-i), padding=(2**(stage-i)+1)//2),
                nn.BatchNorm2d(nb_filter[0]),
                # nn.ReLU(),
            ) for i in range(stage)])
        self.nb_filter = nb_filter
        # building first layer
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.input_size = 256
        self.patch_size = 4
        self.embed_dim = 64

        for j in range (self.block):
            for i in range(self.stage):
                # inp_c = nb_filter[0]*((j-1)*(2**self.stage-1)+(2**(i+1)-1)) # all fore
                if i+j==0:
                    continue
                inp_c = nb_filter[i]*(j)+nb_filter[0]*(2**(i)-1)  # cross
                self.channel_squeeze_list[i][j] = Mask_c(inp_c, inp_c)
                self.node_list[i][j] = self._make_layer(block, inp_c, nb_filter[i], stride=1)
        self.uplayer3_1 = self._make_layer(block, nb_filter[4]+ nb_filter[3], nb_filter[3], stride=1)
        self.uplayer2_1 = self._make_layer(block, nb_filter[3]+ nb_filter[2], nb_filter[2], stride=1)
        # self.decoder = CrossDecoder(num_heads=4)
        self.final_ = nn.Sequential(
            nn.Conv2d (nb_filter[0], nb_filter[0]//4, kernel_size=1),
            nn.BatchNorm2d(nb_filter[0]//4),
            nn.ReLU(),
            nn.Conv2d (nb_filter[0]//4, num_classes, kernel_size=1),
            # nn.BatchNorm2d(num_classes),
        )
        self.decoder = MambaIRv2Light(
                        upscale=4,
                        img_size=64,
                        in_chans=nb_filter[2],
                        out_chans=nb_filter[0],
                        embed_dim=16,
                        d_state=4,
                        depths=[2, 3, 3],
                        num_heads=[4, 4, 4],
                        window_size=16,
                        inner_rank=32,
                        num_tokens=64,
                        convffn_kernel_size=5,
                        img_range=1.,
                        mlp_ratio=1.,
                        upsampler='nearest+conv')
        # self.conv1x1 = nn.ModuleList([nn.ModuleList([]) for _ in range(stage)]) 
        # for i in range(self.stage):
        #     self.conv1x1[i] = nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_4_final = self._make_layer(block, nb_filter[0]*self.stage, nb_filter[0], 1)
        self.final_list = nn.ModuleList([nn.ModuleList([]) for _ in range(stage)])
        for i in range(stage):
            self.final_list[i] = nn.Conv2d(nb_filter[i], num_classes, kernel_size=1)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


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
        batch_num, in_ch, _, _ = input.shape
        # norm_2 = torch.zeros(1, batch_num+1).to(input.device)
        feat_list = [[[] for _ in range(self.block)] for _ in range(self.stage)]
        # self.arch_block = torch.softmax(self.arch_block, dim=-1)
        feat_list[0][0] = self.conv0_0(input)  #  in (b 3 256 256)  x0_0 (b 16 256 256)
        channel_full = torch.tensor([in_ch])
        channel_masked = torch.tensor([in_ch])
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
                feat_list[i][j] = self.node_list[i][j](in_feat) #*mask_c
                channel_masked = torch.cat((channel_masked, torch.tensor([norm_c])))
                channel_full = torch.cat((channel_full, torch.tensor([norm_c_t])))

        feat3_1 = self.uplayer3_1(torch.cat([self.up2(feat_list[4][0]), feat_list[3][0]], 1))
        feat2_1 = self.uplayer2_1(torch.cat([self.up2(feat3_1), feat_list[2][0]], 1))

        # Final_ = self.conv0_4_final(
        #     torch.concat([self.up_list[i](self.conv1x1[i](feat_list[i][self.block-1])) if i>0 else feat_list[i][self.block-1] for i in range(self.stage)], 1)) # Final_x0_4 (b 16 256 256)
        output = [[] for _ in range(self.stage)]
        for i in range(self.stage):
            output[i] = self.final_list[i](self.up_list[i](feat_list[i][self.block-1]))
        # output[self.stage-1] = self.final_list[self.stage-1](encoder_feature)

        encoder_feature = self.conv0_4_final(
            torch.concat([self.down_list[i](feat_list[i][self.block-1]) for i in range(self.stage)], 1)) 
        out = self.decoder(feat2_1)
        # out = self.up4(encoder_feature)
        output.append(self.final_(out))


        # out = self.decoder(feat_list[i][self.block-1] for i in range(self.stage))
        # output = self.final_(out)

        chn_real = torch.tensor(channel_masked).sum()
        chn_org = torch.tensor(channel_full).sum()
        return output, chn_real, chn_org# , maskconv



