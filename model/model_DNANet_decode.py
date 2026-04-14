import torch
import torch.nn as nn


class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x): # 输入输出同尺度
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out
    

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()  # input b 116 10 10
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)       # b 2 58 10 10

    x = torch.transpose(x, 1, 2).contiguous()   # b 58 2 10 10

    # flatten
    x = x.view(batchsize, -1, height, width)     # b 116 10 10

    return x

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, activation='ReLU',norm_layer=nn.BatchNorm2d):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        # assert (self.stride != 1) or (inp == branch_features << 1)
        self.two_braches = (self.stride > 1) or (inp != branch_features << 1)

        if self.two_braches:  # first block in each stage s=2
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),    # depthwise (or padding=1
                norm_layer(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),    # pointwise
                norm_layer(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if self.two_braches else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),    # depthwise (or padding=1
            norm_layer(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),    # pointwise
            norm_layer(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.two_braches:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)  # b 116 10 10
        else:
            x1, x2 = x.chunk(2, dim=1)  # ?? b 116 10 10 -> 2* b 58 10 10   b 464 3 3
            out = torch.cat((x1, self.branch2(x2)), dim=1)  # ?? b 116 10 10
        out = channel_shuffle(out, 2)    # channel shuffle attn
        return out


class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False, learn_loss_mask=False, cam=False):   # [16, 32, 64, 128, 256] [2,2,2,2]
        super(DNANet, self).__init__()
        self.cam =cam
        self.learn_loss_mask = learn_loss_mask
        self.block = block
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])
        # self.conv0_3_final = self._make_layer(block, nb_filter[0]*4, nb_filter[0])
        # self.conv0_2_final = self._make_layer(block, nb_filter[0]*3, nb_filter[0])
        # self.conv0_1_final = self._make_layer(block, nb_filter[0]*2, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

        if self.learn_loss_mask:
            self.lossconv1x1 = nn.Conv2d (nb_filter[0]*2, 1, kernel_size=1)
            self.loss_spattn = nn.Conv2d (2, 1, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)  #  in (b 3 256 256)  x0_0 (b 16 256 256)
        x1_0 = self.conv1_0(self.pool(x0_0))    # x1_0(b 32 128 128)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))    # x0_1 (b 16 256 256)

        x2_0 = self.conv2_0(self.pool(x1_0))    # x2_0 (b 64 64 64)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))    # x1_1 (b 32 128 128)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))  # x0_2 (b 16 256 256)

        x3_0 = self.conv3_0(self.pool(x2_0))    # x3_0 (b 128 32 32)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))    # x2_1 (b 64 64 64)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))  # x1_2 (b 32 128 128)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)) # x0_3 (b 16 256 256)

        x4_0 = self.conv4_0(self.pool(x3_0))    # x4_0 (b 256 16 16)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))    # x1_3 (b 32 128 128)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)) # x0_4 (b 16 256 256)

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1)) # Final_x0_4 (b 16 256 256)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            if self.cam:
                return [x4_0, x3_1, x2_2, x1_3, x0_4, output1, output2, output3, output4]
            return [output1, output2, output3, output4] # , maskconv
        else:
            output = self.final(Final_x0_4)
            return output   # , maskconv
        
        # 从第一级得到四维输出
        # Final_x0_4 = self.conv0_4_final(
        #     torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        # Final_x0_3 = self.conv0_3_final(
        #     torch.cat([self.up_8(self.conv0_3_1x1(x3_1)),
        #                self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))
        
        # Final_x0_2 = self.conv0_2_final(
        #     torch.cat([self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))
        
        # Final_x0_1 = self.conv0_1_final(
        #     torch.cat([self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))
        # if self.deep_supervision:
        #     output1 = self.final1(Final_x0_1)
        #     output2 = self.final2(Final_x0_2)
        #     output3 = self.final3(Final_x0_3)
        #     output4 = self.final4(Final_x0_4)

        #     return [output1, output2, output3, output4]
        # else:
        #     output = self.final(Final_x0_4)
        #     return output




