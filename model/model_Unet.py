import torch
import torch.nn as nn
from torch.autograd import Variable
import thop
import torch.nn.functional as F
from model.mix_blocks import *
from model.attentions import *
from model.model_FANet import ACmix
import einops as ei

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False, learn_loss_mask=False, cam=False):   # [16, 32, 64, 128, 256] [2,2,2,2]
        super(UNet, self).__init__()
        self.cam=cam
        self.learn_loss_mask = learn_loss_mask
        self.block = block
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.max_pool2 = nn.MaxPool2d(2, stride=2)
        self.max_pool4 = nn.MaxPool2d(3,stride=4)
        self.max_pool8 = nn.MaxPool2d(3,stride=8)
        self.max_pool16 = nn.MaxPool2d(3,stride=16)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0], num_blocks[0], stride=1)
        # self.vitdown0 = ViTDown_Block(nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0], stride=2)
        # self.vitdown1 = ViTDown_Block(nb_filter[1], nb_filter[1])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1], stride=2)
        # self.vitdown2 = ViTDown_Block(nb_filter[2], nb_filter[2])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2], stride=2)
        # self.vitdown3 = ViTDown_Block(nb_filter[3], nb_filter[3])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3], stride=2)# nb_filter[0]+nb_filter[1]+nb_filter[2]+

        # self.neck0 = BottleneckCSP(nb_filter[0], nb_filter[0])
        # self.neck1 = BottleneckCSP(nb_filter[1], nb_filter[1])
        # self.neck2 = BottleneckCSP(nb_filter[2], nb_filter[2])
        # self.neck3 = BottleneckCSP(nb_filter[3], nb_filter[3])

        self.neck0 = BottleneckCSP(nb_filter[0], nb_filter[0])
        self.neck1 = BottleneckCSP(nb_filter[1], nb_filter[1])
        self.neck2 = BottleneckCSP(nb_filter[2], nb_filter[2])
        self.neck3 = BottleneckCSP(nb_filter[3], nb_filter[3])

        self.conv3_1 = TOAA(nb_filter[3], nb_filter[4], nb_filter[3])
        self.conv2_2 = TOAA(nb_filter[2], nb_filter[3], nb_filter[2])
        # self.conv2_1 = self._make_layer(TOAA, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv1_3 = TOAA(nb_filter[1],  nb_filter[2], nb_filter[1])
        # self.conv1_2 = self._make_layer(TOAA, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv0_4 = TOAA(nb_filter[0], nb_filter[1], nb_filter[0])
        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])
        self.acmix = ACmix(nb_filter[0], nb_filter[0], (8, 8), 4)
        # self.se = SEblock(nb_filter[0]*2, nb_filter[0])

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

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1, stride=1):
        layers = []
        # if input_channels > 4:
        #     layers.append(SPG(input_channels, input_channels))
        layers.append(block(input_channels, output_channels, stride))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def get_gradient(self, image_tensor):
        # Sobel operator for edge detection
        B,C,W,H = image_tensor.shape
        sobel_operator_x = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]])
        sobel_operator_y = torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]])

        sobel_operator_x = sobel_operator_x.repeat(C,C,1,1).cuda()
        sobel_operator_y = sobel_operator_y.repeat(C,C,1,1).cuda()

        # assuming image_tensor is your [b,c,w,h] tensor
        gradient_x = F.conv2d(image_tensor, sobel_operator_x, padding=1)
        gradient_y = F.conv2d(image_tensor, sobel_operator_y, padding=1)
        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        return gradient_magnitude

    def forward(self, input):
        B,C,W,H = input.size()
        x0_0 = self.conv0_0(input)  #  in (b 3 256 256)  x0_0 (b 16 256 256)
        x1_0 = self.conv1_0(x0_0)   # x1_0(b 32 128 128)
        x2_0 = self.conv2_0(x1_0)   # x2_0 (b 64 64 64)
        x3_0 = self.conv3_0(x2_0)    # x3_0 (b 128 32 32)
        x4_0 = self.conv4_0(x3_0) # x4_0 (b 256 16 16)
        # out_4 = torch.cat([ei.rearrange(self.vit2(x3_0), "b (x y) c -> b c x y", x=W//16, y=H//16),
        #                    ei.rearrange(self.vit3(x2_0), "b (x y) c -> b c x y", x=W//16, y=H//16),
        #                    ei.rearrange(self.vit4(x1_0), "b (x y) c -> b c x y", x=W//16, y=H//16),
        #                    ei.rearrange(self.vit5(x0_0), "b (x y) c -> b c x y", x=W//16, y=H//16), x4_0], 1) # ([b, 512, 16, 16])

        # x4_0 = self.conv4_0(torch.cat([self.vitdown0(x0_0), self.vitdown1(x1_0), self.vitdown2(x2_0), self.vitdown3(x3_0)], 1))

        # grad = self.get_gradient(input)
        # g0_0 = self.conv0_0(grad)
        # g1_0 = self.conv1_0(g0_0)
        # g2_0 = self.conv2_0(g1_0)
        # g3_0 = self.conv3_0(g2_0)

        x0_0 = self.neck0(x0_0)
        x1_0 = self.neck1(x1_0)
        x2_0 = self.neck2(x2_0)
        x3_0 = self.neck3(x3_0)

        x3_1 = self.conv3_1(low=x3_0, high=self.up(x4_0))    # -self.down(x2_1)
        x2_2 = self.conv2_2(low=x2_0, high=self.up(x3_1))    # -x2_1, self.down(x1_2)
        # x2_1 = self.conv2_1(x2_0,x2_2)
        x1_3 = self.conv1_3(low=x1_0, high=self.up(x2_2))    # x1_3 (b 32 128 128)  -x1_1, x1_2, self.down(x0_3)
        x0_4 = self.conv0_4(low=x0_0, high=self.up(x1_3)) # x0_4 (b 16 256 256)  # -x0_1, x0_2, x0_3,
        # Final_x0_4 = self.conv0_4_final(
        #     torch.cat([self.up_8(self.conv0_3_1x1(x3_0)),
        #                self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1)) # Final_x0_4 (b 16 256 256)
        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), self.acmix(x0_4)], 1)) # Final_x0_4 (b 16 256 256)
        # out = self.se(torch.cat([Final_x0_4, x0_0], 1))

        output = self.final(Final_x0_4)
        # if self.cam:
        #     return [x4_0, x3_1, x2_2, x1_3, x0_4, Final_x0_4, output]
        return output   # , maskconv





