import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

BLOCKS_NAME = [
    'none',
    'res_cbam',
    'shuffle',
    'rfb',
]

Blocks = {
    'none': lambda cin, cout, b_num: Identity(cin, cout),
    'res_cbam': lambda cin, cout, b_num: _make_layer(Res_CBAM_block, cin, cout, b_num),
    'shuffle': lambda cin, cout, b_num: _make_layer(ShuffleV2Block, cin, cout, b_num),
    'rfb': lambda cin, cout, b_num: _make_layer(BasicRFB_a, cin, cout, b_num),

}


class weight_concat(nn.Module):
    def __init__(self, layer_chn, k) -> None:
        super().__init__()
        self.layer_chn = layer_chn
        self.upconv1x1 = nn.Sequential(
            nn.Conv2d(layer_chn // 2, layer_chn, 1),
            nn.BatchNorm2d(layer_chn),
            nn.ReLU(),
        )
        self.downconv1x1 = nn.Sequential(
            nn.Conv2d(layer_chn * 2, layer_chn, 1),
            nn.BatchNorm2d(layer_chn),
            nn.ReLU(),
        )
        self.k = k
        # self.popup_scores = nn.Parameter(torch.Tensor((opnum,1)))
        # nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

    def binary_encoding(self, n):
        size = 2 ** n
        matrix = torch.zeros((size, n))
        for i in range(0, size):
            binary = bin(i)[2:].zfill(n)
            matrix[i] = torch.tensor([float(bit) for bit in binary])
        return matrix[1:]

    def plain_sigmoid(self, feats, arch_connect, masked=False):
        '''
        feats: list of inputs; l = len(feats)
        arch_connect: 2**l-1
        '''
        scores = torch.sigmoid(arch_connect)
        if masked:
            arch_c = ConnectionBinaryChoice.apply(scores, self.k)
        else:
            arch_c = scores
        # for i, feat in enumerate(feats):
        #     if feat.shape[1] == self.layer_chn // 2:
        #         feat = self.upconv1x1(feat)
        #     elif feat.shape[1] == self.layer_chn * 2:
        #         feat = self.downconv1x1(feat)
        #     featlist.append(feat * arch_c[i])
        for i in range(len(feats)):
            if feats[i].shape[1] == self.layer_chn // 2:
                feats[i] = self.upconv1x1(feats[i])
            elif feats[i].shape[1] == self.layer_chn * 2:
                feats[i] = self.downconv1x1(feats[i])
        arch_c = arch_c.expand(self.layer_chn, len(feats)).permute(1,0).flatten().unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return torch.concat(feats, dim=1)*arch_c

    def forward(self, feats, arch_connect, pruned=False):
        '''
        feats: list of inputs; l = len(feats)
        arch_connect: 2**l-1
        '''
        n = len(feats)
        if pruned:
            scores = torch.softmax(arch_connect, dim=0)
            arch_c = ConnectionBinaryChoice.apply(scores, self.k)
            idx2bit = arch_c @ self.binary_encoding(n).cuda()
        else:
            idx2bit = self.binary_encoding(n)[-1].cuda()
        for i in range(n):
            if feats[i].shape[1] == self.layer_chn // 2:
                feats[i] = self.upconv1x1(feats[i])
            elif feats[i].shape[1] == self.layer_chn * 2:
                feats[i] = self.downconv1x1(feats[i])
        arch_c = idx2bit.expand(self.layer_chn, len(feats)).permute(1,0).flatten().unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return torch.concat(feats, dim=1)*arch_c


class ConnectionBinaryChoice(autograd.Function):
    @staticmethod
    def forward(ctx, arch_connect, k=1):

        idx = int(torch.argmax(arch_connect))
        flat_out = arch_connect.clone()
        flat_out[:] = 0.
        flat_out[idx] = 1.

        return flat_out

    def score_flatten(self, arch_connect, k=1):
        scores = arch_connect.clone()
        _, idx = scores.flatten().sort()
        # j = int((1 - k) * scores.numel())
        j = k
        assert k < len(scores)
        flat_out = scores.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return flat_out

    @staticmethod
    def backward(ctx, g):
        return g, None



def conv1x1prune(in_planes, out_planes, conv_layer, stride=1, k=1.0):
    """1x1 convolution"""
    a = conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, dilation=1)
    if conv_layer == SubnetConvChannel:
        a.set_prune_rate(k=k)
    return a



class GetSubnetChannel(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, p=1):  # binarization
        # Get the subnetwork by sorting the scores and using the top k%

        score_L1_norm = torch.norm(torch.norm(scores, p=p, dim=[2, 3]), p=p, dim=0)
        _, idx = score_L1_norm.sort()
        j = int((1 - k) * scores.shape[1])

        # flat_out and out access the same memory.
        out = scores.clone()
        out[:, idx[:j], :, :] = 0
        out[:, idx[j:], :, :] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None


class SubnetConvChannel(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            k=1.0
    ):
        super(SubnetConvChannel, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = nn.Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        # self.weight.requires_grad = False
        # if self.bias is not None:
        #     self.bias.requires_grad = False
        self.w = 0
        self.p = 1
        self.k = k

    def set_prune_rate(self, k):
        self.k = k
        self.p = 2

    def forward(self, x):
        adj = GetSubnetChannel.apply(self.popup_scores.abs(), self.k, self.p)

        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class BlockBinaryChoice(autograd.Function):
    @staticmethod
    def forward(ctx, arch_block):
        max_values, indexmax = torch.max(arch_block, dim=0)
        # mask = torch.zeros_like(arch_block)
        mask = arch_block.clone()
        mask[indexmax] = 1.
        mask[mask!=1] = 0.
        # arch_block = mask
        # max_values, indexmax = torch.max(arch_block, dim=-1)
        # eps = 1e-5
        # out = torch.sign(arch_block - max_values.resize(15, 1) + eps)
        # out = (out + 1) / 2
        return mask

    @staticmethod
    def backward(ctx, g):
        return g


class MixBlock(nn.Module):
    def __init__(self, cin, cout, b_num=1):
        super().__init__()
        self._blocks = nn.ModuleList()
        for option in BLOCKS_NAME:
            block = Blocks[option](cin, cout, b_num)
            self._blocks.append(block)
            
    def forward(self, x, weights, masked=False, test=False):
        wei_softmax = torch.softmax(weights, dim=0)
        wei_binary = BlockBinaryChoice.apply(wei_softmax)
        wei_ = wei_binary if masked else wei_softmax
        if test:
            return self._blocks[torch.argmax(wei_)](x)
        else:
            return sum(w * op(x) for w, op in zip(wei_, self._blocks))

        # if max(weights) == 1:
        #     return self._blocks[torch.argmax(wei_binary)](x)
        # else:
        #     return sum(w * op(x) for w, op in zip(wei_binary, self._blocks))
        # return sum(w * op(x) for w, op in zip(weights, self._blocks))

def _make_layer(block, input_channels,  output_channels, num_blocks=1):
    layers = []
    layers.append(block(input_channels, output_channels))
    for i in range(num_blocks-1):
        layers.append(block(output_channels, output_channels))
    return nn.Sequential(*layers)

class Identity(nn.Module):

  def __init__(self, cin, cout):
    super(Identity, self).__init__()
    if cin == cout:
        self.op = nn.Identity()
    else:
        self.op = nn.Sequential(
            nn.Conv2d(cin, cout, 1),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )

  def forward(self, x):
    return self.op(x)

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
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
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
    def __init__(self, inp, oup, stride=1, padding=0, dilation=1, activation='ReLU',norm_layer=nn.BatchNorm2d):
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
    

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups = in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 1.0):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4 if in_planes>=4 else 1

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, (inter_planes//2 if inter_planes>=2 else 1), kernel_size=1, stride=1),
                BasicConv((inter_planes//2 if inter_planes>=2 else 1), (inter_planes//4 if inter_planes>=4 else 1)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4 if inter_planes>=4 else 1)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

        if stride != 1 or out_planes != in_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_planes))
        else:
            self.shortcut = None

    def forward(self,x):
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x
        x0 = self.branch0(x)    # b c/4 w h
        x1 = self.branch1(x)    # b c/4 w h 
        x2 = self.branch2(x)    # b c/4 w h
        x3 = self.branch3(x)    # b c/4 w h

        out = torch.cat((x0,x1,x2,x3),1)    # b c w h
        out = self.ConvLinear(out)
        out = out*self.scale + residual
        out = self.relu(out)

        return out
    

def model_info(model, input_size):
    input = torch.rand(input_size)
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    # print('thop| gflops:%.2fG  params:%.2fM'%(flops/ 1E9, params/ 1e6))
    print(f"model info| summary: {len(list(model.modules()))} layers, {n_p /1E6}M parameters, {n_g /1E6}M gradients {flops/1E9}")
    return n_p, flops

if __name__ == '__main__':
    cin = [3, 16, 32]
    cout = [16,32]
    b_num = 2
    # size = [128, 256]
    for inp in cin:
        for outp in cout:
            print(inp, outp)
            model1 = Res_CBAM_block(inp, outp)
            model2 = Res_CBAM_block(outp, outp)
            model_info(model1, (2, inp, 128 ,128))
            model_info(model2, (2, outp, 128 ,128))
