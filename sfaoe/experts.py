"""空间专家、小波专家及其残差混合块。"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data
from timm.models.layers import DropPath
from sfaoe.routing import SpatialSparseMoE



def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return (dec_filters, rec_filters)


def wavelet_transform(x, filters):
    (b, c, h, w) = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    (b, c, _, h_half, w_half) = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class ContextWaveletConv2d(nn.Module):
    """Context expert: filter all LL/LH/HL/HH bands and reconstruct with IWT.

    This keeps the original two-level context computation. No separate detail
    enhancement path is constructed. Odd dimensions are padded then cropped.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=2,
        wt_type="db1",
    ):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        (self.wt_filter, self.iwt_filter) = create_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding="same",
                    stride=1,
                    dilation=1,
                    groups=in_channels * 4,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [
                _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
                for _ in range(self.wt_levels)
            ]
        )
        if self.stride > 1:
            self.stride_filter = nn.Parameter(
                torch.ones(in_channels, 1, 1, 1), requires_grad=False
            )
            self.do_stride = lambda x_in: F.conv2d(
                x_in,
                self.stride_filter,
                bias=None,
                stride=self.stride,
                groups=in_channels,
            )
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        # 每层仅继续分解 LL，但四个子带都参与可学习滤波与重建。
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if curr_shape[2] % 2 > 0 or curr_shape[3] % 2 > 0:
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(
                shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4]
            )
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
        next_x_ll = 0
        # 从最粗尺度逐级 IWT，并按保存的形状裁掉补齐边界。
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        x = x + x_tag
        if self.do_stride is not None:
            x = self.do_stride(x)
        return x


# Legacy import name; this alias only implements the context expert.
MBWTConv2d = ContextWaveletConv2d


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module(
            "dwconv3x3",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
        )
        self.add_module("bn1", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "dwconv1x1",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=out_channels,
                bias=False,
            ),
        )
        self.add_module("bn2", nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        (dwconv3x3, bn1, relu, dwconv1x1, bn2) = self._modules.values()
        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = (
            bn1.bias
            - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        )
        fused_dwconv3x3 = nn.Conv2d(
            w1.size(1) * dwconv3x3.groups,
            w1.size(0),
            w1.shape[2:],
            stride=dwconv3x3.stride,
            padding=dwconv3x3.padding,
            dilation=dwconv3x3.dilation,
            groups=dwconv3x3.groups,
            device=dwconv3x3.weight.device,
        )
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = (
            bn2.bias
            - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        )
        fused_dwconv1x1 = nn.Conv2d(
            w2.size(1) * dwconv1x1.groups,
            w2.size(0),
            w2.shape[2:],
            stride=dwconv1x1.stride,
            padding=dwconv1x1.padding,
            dilation=dwconv1x1.dilation,
            groups=dwconv1x1.groups,
            device=dwconv1x1.weight.device,
        )
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1
    ):
        super().__init__()
        self.add_module(
            "c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        self.add_module("bn", torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        (c, bn) = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module("bn", torch.nn.BatchNorm1d(a))
        self.add_module("l", torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        (bn, l) = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = (
            bn.bias
            - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        )
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, 0.25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.0):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return (
                x
                + self.m(x)
                * torch.rand(x.size(0), 1, 1, 1, device=x.device)
                .ge_(self.drop)
                .div(1 - self.drop)
                .detach()
            )
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = n // 16 * 16
        upper_multiple = lower_multiple + 16
        if n - lower_multiple < upper_multiple - n:
            return lower_multiple
        else:
            return upper_multiple


def channel_shuffle(x, groups):
    (batchsize, num_channels, height, width) = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicSepConv(nn.Module):
    def __init__(
        self,
        in_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_planes,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB_a(nn.Module):
    """多分支空间专家，融合局部、方向与膨胀卷积响应。"""

    def __init__(self, in_planes, out_planes, stride=1, scale=1.0, dilations=None):
        super(BasicRFB_a, self).__init__()
        if dilations is None:
            dilations = [1, 2, 2, 3]
        (d0, d1, d2, d3) = dilations
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4 if in_planes >= 4 else 1
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicSepConv(
                inter_planes,
                kernel_size=3,
                stride=1,
                padding=d0,
                dilation=d0,
                relu=False,
            ),
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(
                inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)
            ),
            BasicSepConv(
                inter_planes,
                kernel_size=3,
                stride=1,
                padding=d1,
                dilation=d1,
                relu=False,
            ),
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(
                inter_planes,
                inter_planes,
                kernel_size=(1, 3),
                stride=stride,
                padding=(0, 1),
            ),
            BasicSepConv(
                inter_planes,
                kernel_size=3,
                stride=1,
                padding=d2,
                dilation=d2,
                relu=False,
            ),
        )
        self.branch3 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes // 2 if inter_planes >= 2 else 1,
                kernel_size=1,
                stride=1,
            ),
            BasicConv(
                inter_planes // 2 if inter_planes >= 2 else 1,
                (inter_planes // 4 if inter_planes >= 4 else 1) * 3,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
            ),
            BasicConv(
                (inter_planes // 4 if inter_planes >= 4 else 1) * 3,
                inter_planes,
                kernel_size=(3, 1),
                stride=stride,
                padding=(1, 0),
            ),
            BasicSepConv(
                inter_planes,
                kernel_size=3,
                stride=1,
                padding=d3,
                dilation=d3,
                relu=False,
            ),
        )
        self.ConvLinear = BasicConv(
            4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False
        )
        self.relu = nn.ReLU(inplace=False)
        if stride != 1 or out_planes != in_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        out = out * self.scale + residual
        out = self.relu(out)
        return out


class MobileMambaModule(torch.nn.Module):
    """Fixed expert order: context, multireceptive spatial, Identity.

    spacial_moe and local_op retain their historical checkpoint key names.
    global_ratio/local_ratio remain accepted for the backbone call signature;
    each expert operates on all channels, with sparse inputs by default.
    """

    def __init__(
        self, dim, global_ratio=0.25, local_ratio=0.25, kernels=3,
        global_type="context", layer=2, dilations=None, noise_scale=0.2,
        patch_size=1, top_k=2, input_routing="sparse",
    ):
        super().__init__()
        if global_type != "context":
            raise ValueError("This branch supports only the context expert")
        if not 1 <= int(top_k) <= 3:
            raise ValueError("top_k must be between 1 and 3")
        self.dim = dim
        self.local_op = BasicRFB_a(dim, dim, dilations=dilations)
        context = ContextWaveletConv2d(dim, dim, kernels, wt_levels=2)
        self.global_op = [context]
        self.expert_names = ("context", "spatial", "identity")
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(dim, dim, bn_weight_init=0))
        experts = nn.ModuleList([context, self.local_op, nn.Identity()])
        self.spacial_moe = SpatialSparseMoE(
            dim, 3, int(top_k), dim, experts, noise_scale=noise_scale,
            patch_size=patch_size, input_routing=input_routing,
        )

    def forward(self, x):
        return self.proj(self.spacial_moe(x))


class MobileMambaBlockWindow(torch.nn.Module):
    def __init__(
        self,
        dim,
        global_ratio=0.25,
        local_ratio=0.25,
        kernels=5,
        ssm_ratio=1,
        forward_type="v052d",
        layer=2,
        dilations=None,
        noise_scale=0.2,
        patch_size=1,
        top_k=2,
        input_routing="sparse",
    ):
        super().__init__()
        self.dim = dim
        self.attn = MobileMambaModule(
            dim,
            global_ratio=global_ratio,
            local_ratio=local_ratio,
            kernels=kernels,
            global_type="context",
            layer=layer,
            dilations=dilations,
            noise_scale=noise_scale,
            patch_size=patch_size,
            top_k=top_k,
            input_routing=input_routing,
        )

    def forward(self, x):
        x = self.attn(x)
        return x


class MobileMambaBlock(torch.nn.Module):
    def __init__(
        self,
        type,
        ed,
        global_ratio=0.25,
        local_ratio=0.25,
        kernels=5,
        drop_path=0.0,
        has_skip=True,
        ssm_ratio=1,
        forward_type="v052d",
        layer=2,
        dilations=None,
        noise_scale=0.2,
        patch_size=1,
        top_k=2,
        input_routing="sparse",
    ):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.0))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))
        if type == "s":
            self.mixer = Residual(
                MobileMambaBlockWindow(
                    ed,
                    global_ratio=global_ratio,
                    local_ratio=local_ratio,
                    kernels=kernels,
                    ssm_ratio=ssm_ratio,
                    forward_type=forward_type,
                    layer=layer,
                    dilations=dilations,
                    noise_scale=noise_scale,
                    patch_size=patch_size,
                    top_k=top_k,
                    input_routing=input_routing,
                )
            )
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.0))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))
        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = shortcut + self.drop_path(x) if self.has_skip else x
        return x
