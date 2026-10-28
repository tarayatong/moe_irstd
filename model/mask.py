"""旧路由接口与实验变体；主路径由 sfaoe.routing 实现。"""

from sfaoe.routing import Expert, SpatialNoisyTopkRouter, SpatialSparseMoE
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class GumbelSoftmax(nn.Module):
    """
    gumbel softmax gate.
    """

    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def gumbel_sample(self, template_tensor, eps=1e-08):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor + eps) - torch.log(
            1 - uniform_samples_tensor + eps
        )
        return gumble_samples_tensor

    def gumbel_softmax(self, logits):
        """Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.eps)
        return (soft_samples, logits)

    def forward(self, logits):
        if not self.training:
            out_hard = (logits >= 0).float()
            return out_hard
        (out_soft, prob_soft) = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard


class Mask_s(nn.Module):
    """
    Attention Mask spatial.
    """

    def __init__(self, h, w, planes, block_w, block_h, eps=0.66667, bias=-1, **kwargs):
        super(Mask_s, self).__init__()
        (self.width, self.height, self.channel) = (w, h, planes)
        (self.mask_h, self.mask_w) = (
            int(np.ceil(h / block_h)),
            int(np.ceil(w / block_w)),
        )
        self.eleNum_s = torch.Tensor([self.mask_h * self.mask_w])
        self.atten_s = nn.Conv2d(
            planes, 1, kernel_size=3, stride=1, bias=bias >= 0, padding=1
        )
        if bias >= 0:
            nn.init.constant_(self.atten_s.bias, bias)
        self.gate_s = GumbelSoftmax(eps=eps)
        self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2, 3))

    def forward(self, x):
        (batch, channel, height, width) = x.size()
        input_ds = F.adaptive_avg_pool2d(
            input=x, output_size=(self.mask_h, self.mask_w)
        )
        s_in = self.atten_s(input_ds)
        mask_s = self.gate_s(s_in)
        norm = self.norm(mask_s)
        norm_t = self.eleNum_s.to(x.device)
        return (mask_s, norm, norm_t)

    def get_flops(self):
        flops = self.mask_h * self.mask_w * self.channel * 9
        return flops


class Mask_c(nn.Module):
    """
    Attention Mask.
    """

    def __init__(
        self, inplanes, outplanes, fc_reduction=4, eps=0.66667, bias=-1, **kwargs
    ):
        super(Mask_c, self).__init__()
        self.bottleneck = inplanes // fc_reduction
        (self.inplanes, self.outplanes) = (inplanes, outplanes)
        self.eleNum_c = torch.Tensor([outplanes])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.atten_c = nn.Sequential(
            nn.Conv2d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.bottleneck, outplanes, kernel_size=1, stride=1, bias=bias >= 0
            ),
        )
        if bias >= 0:
            nn.init.constant_(self.atten_c[3].bias, bias)
        self.gate_c = GumbelSoftmax(eps=eps)
        self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2, 3))

    def forward(self, x):
        (batch, channel, _, _) = x.size()
        context = self.avg_pool(x)
        c_in = self.atten_c(context)
        mask_c = self.gate_c(c_in)
        norm = self.norm(mask_c)
        norm_t = self.eleNum_c.to(x.device)
        return (mask_c, int(norm.mean().item()), int(norm_t.mean().item()))

    def get_flops(self):
        flops = self.inplanes * self.bottleneck + self.bottleneck * self.outplanes
        return flops


class Sequential_DG(nn.Sequential):
    def __init__(self, layers):
        super(Sequential_DG, self).__init__(*layers)
        self._module_num = len(layers)

    def forward(self, input):
        (x, mask_c) = input
        i = 0
        for module in self._modules.values():
            if i == self._module_num - 1:
                x = x * mask_c
            x = module(x)
        i += 1
        return x


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise * 0.2
        (top_k_logits, indices) = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return (router_output, indices)


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, out_channels):
        super(SparseMoE, self).__init__()
        self.output_dim = out_channels
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(n_embed, out_channels) for _ in range(num_experts)]
        )
        self.top_k = top_k

    def forward(self, x):
        (batch_size, seq_len, input_dim) = x.size()
        (gating_output, indices) = self.router(x)
        final_output = torch.zeros(batch_size, seq_len, self.output_dim).to(x.device)
        flat_x = x.reshape(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output


class MoELinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, top_k):
        super(MoELinearLayer, self).__init__()
        self.moe = SparseMoE(in_channels, num_experts, top_k, out_channels)

    def forward(self, x):
        (batch_size, in_channels, height, width) = x.size()
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, -1, in_channels)
        x_transformed = self.moe(x_flat)
        x_out = x_transformed.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        return x_out


class ChannelRouter(nn.Module):
    def __init__(self, in_channels, num_experts, top_k, out_channels, experts=None):
        super(ChannelRouter, self).__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.topk = top_k
        self.routing_layer = nn.Conv2d(
            in_channels, in_channels * num_experts, kernel_size=1, groups=in_channels
        )
        self.experts = experts

    def forward(self, x):
        (B, C, H, W) = x.shape
        raw_scores = (
            self.routing_layer(x).view(B, self.num_experts, C, H, W).mean(dim=[3, 4])
        )
        raw_scores = raw_scores.permute(0, 2, 1)
        if self.topk > 0:
            topk_mask = torch.zeros_like(raw_scores)
            topk_idx = torch.topk(raw_scores, self.topk, dim=-1).indices
            topk_mask.scatter_(-1, topk_idx, raw_scores.gather(-1, topk_idx))
            final_output = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                weight = topk_mask[:, :, i].unsqueeze(-1).unsqueeze(-1)
                out = expert(x)
                final_output += out * weight
        else:
            final_output = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                weight = raw_scores[:, :, i].unsqueeze(-1).unsqueeze(-1)
                out = expert(x)
                final_output += out * weight
        return final_output


class GroupedChannelSelection(nn.Module):
    def __init__(self, in_channels, out_channles, groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True),
        )
        self.groups = groups
        channels_per_group = out_channles // groups
        self.group_selection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channles, groups, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        group_weights = self.group_selection(x)
        channel_weights = group_weights.repeat_interleave(
            x.size(1) // self.groups, dim=1
        )
        return channel_weights


class SpatialExpert(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.net(x)


class SpatialSparse(nn.Module):
    def __init__(self, in_channels, num_experts, top_k, out_channels, experts=None):
        super().__init__()
        if experts == None:
            self.experts = nn.ModuleList(
                [SpatialExpert(in_channels, out_channels) for _ in range(num_experts)]
            )
        else:
            self.experts = experts
        self.top_k = top_k
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.router = nn.Parameter(torch.rand(in_channels, num_experts))
        self.gumble_softmax = GumbelSoftmax()

    def forward(self, x):
        (B, C, H, W) = x.shape
        routing_weights = self.gumble_softmax(self.router)
        routing_weights = routing_weights.unsqueeze(0).expand(B, C, self.num_experts)
        final_output = torch.zeros(B, self.out_channels, H, W, device=x.device)
        for i, expert in enumerate(self.experts):
            expert_mask = routing_weights[:, :, i]
            masked_input = x * expert_mask.unsqueeze(-1).unsqueeze(-1)
            expert_output = expert(masked_input)
            final_output = final_output + expert_output
        return final_output


class SpatialMoELayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_experts, top_k, input_routing="full"
    ):
        super().__init__()
        self.moe = SpatialSparseMoE(
            in_channels, num_experts, top_k, out_channels, input_routing=input_routing
        )

    def forward(self, x):
        return self.moe(x)


def test_moe_linear_layer():
    torch.manual_seed(42)
    batch_size = 2
    in_channels = 64
    height = 32
    width = 32
    out_channels = 128
    num_experts = 4
    top_k = 2
    model = MoELinearLayer(in_channels, out_channels, num_experts, top_k)
    x = torch.randn(batch_size, in_channels, height, width)
    output = model(x)
    expected_shape = (batch_size, out_channels, height, width)
    assert output.shape == expected_shape, (
        f"输出形状错误: 期望 {expected_shape}, 得到 {output.shape}"
    )
    assert not torch.isnan(output).any(), "输出包含 NaN 值"
    assert not torch.isinf(output).any(), "输出包含 Inf 值"
    output.sum().backward()
    for param in model.parameters():
        assert param.grad is not None, "某些参数没有梯度"
        assert not torch.isnan(param.grad).any(), "梯度包含 NaN 值"
        assert not torch.isinf(param.grad).any(), "梯度包含 Inf 值"
    print("基本功能测试通过")
    test_sizes = [(1, 64, 16, 16), (4, 64, 64, 64), (8, 64, 8, 8)]
    for size in test_sizes:
        x = torch.randn(*size)
        output = model(x)
        expected_shape = (size[0], out_channels, size[2], size[3])
        assert output.shape == expected_shape, (
            f"输入尺寸 {size} 测试失败: 期望输出形状 {expected_shape}, 得到 {output.shape}"
        )
    print("不同输入尺寸测试通过")
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(batch_size, in_channels, height, width).cuda()
        output = model(x)
        assert output.device.type == "cuda", "GPU 测试失败"
        print("GPU 兼容性测试通过")
    model.eval()
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
        assert torch.allclose(output1, output2), "模型输出不一致"
    print("行为一致性测试通过")


def test_spatial_moe():
    batch_size = 2
    in_channels = 64
    out_channels = 128
    height = 32
    width = 32
    num_experts = 4
    top_k = 2
    model = SpatialMoELayer(in_channels, out_channels, num_experts, top_k)
    x = torch.randn(batch_size, in_channels, height, width)
    output = model(x)
    expected_shape = (batch_size, out_channels, height, width)
    assert output.shape == expected_shape
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("测试通过!")


if __name__ == "__main__":
    test_spatial_moe()
