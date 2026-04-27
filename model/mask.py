import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class GumbelSoftmax(nn.Module):
    '''
        gumbel softmax gate.
    '''
    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
    
    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor+eps)-torch.log(
                                          1-uniform_samples_tensor+eps)
        return gumble_samples_tensor
    
    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits
    
    def forward(self, logits):
        if not self.training:
            out_hard = (logits>=0).float()
            return out_hard
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard


class Mask_s(nn.Module):
    '''
        Attention Mask spatial.
    '''
    def __init__(self, h, w, planes, block_w, block_h, eps=0.66667,
                 bias=-1, **kwargs):
        super(Mask_s, self).__init__()
        # Parameter
        self.width, self.height, self.channel = w, h, planes
        self.mask_h, self.mask_w = int(np.ceil(h / block_h)), int(np.ceil(w / block_w))
        self.eleNum_s = torch.Tensor([self.mask_h*self.mask_w])
        # spatial attention
        self.atten_s = nn.Conv2d(planes, 1, kernel_size=3, stride=1, bias=bias>=0, padding=1)
        if bias>=0:
            nn.init.constant_(self.atten_s.bias, bias)
        # Gate
        self.gate_s = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
    
    def forward(self, x):
        batch, channel, height, width = x.size()
        # Pooling
        input_ds = F.adaptive_avg_pool2d(input=x, output_size=(self.mask_h, self.mask_w))
        # spatial attention
        s_in = self.atten_s(input_ds) # [N, 1, h, w]
        # spatial gate
        mask_s = self.gate_s(s_in) # [N, 1, h, w]
        # norm
        norm = self.norm(mask_s)
        norm_t = self.eleNum_s.to(x.device)
        return mask_s, norm, norm_t
    
    def get_flops(self):
        flops = self.mask_h * self.mask_w * self.channel * 9
        return flops


class Mask_c(nn.Module):
    '''
        Attention Mask.
    '''
    def __init__(self, inplanes, outplanes, fc_reduction=4, eps=0.66667, bias=-1, **kwargs):
        super(Mask_c, self).__init__()
        # Parameter
        self.bottleneck = inplanes // fc_reduction 
        self.inplanes, self.outplanes = inplanes, outplanes
        self.eleNum_c = torch.Tensor([outplanes])
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.atten_c = nn.Sequential(
            nn.Conv2d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck, outplanes, kernel_size=1, stride=1, bias=bias>=0),
        )
        if bias>=0:
            # self.atten_c[3].bias.data = self.atten_c[3].bias.data.cuda()
            nn.init.constant_(self.atten_c[3].bias, bias)
        # Gate
        self.gate_c = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
    
    def forward(self, x):
        batch, channel, _, _ = x.size()
        context = self.avg_pool(x) # [N, C, 1, 1] 
        # transform
        c_in = self.atten_c(context) # [N, C_out, 1, 1]
        # channel gate
        mask_c = self.gate_c(c_in) # [N, C_out, 1, 1]
        # norm
        norm = self.norm(mask_c)
        norm_t = self.eleNum_c.to(x.device)
        return mask_c, int(norm.mean().item()), int(norm_t.mean().item())
    
    def get_flops(self):
        flops = self.inplanes * self.bottleneck + self.bottleneck * self.outplanes
        return flops


class Sequential_DG(nn.Sequential):
    def __init__(self, layers):
        super(Sequential_DG, self).__init__(*layers)
        self._module_num = len(layers)

    def forward(self, input):
        x, mask_c = input
        i = 0
        for module in self._modules.values():
            if i == self._module_num-1:
                x = x * mask_c
            x = module(x)
        i += 1
        return x
    

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise*0.2

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class Expert(nn.Module):
    def __init__(self, n_embd, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(n_embd, 2 * n_embd),
            # nn.ReLU(),
            nn.Conv2d(n_embd, out_channels, 3, 1, 1),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, out_channels):
        super(SparseMoE, self).__init__()
        self.output_dim = out_channels
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed, out_channels) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        # 1. 输入进入router得到两个输出
        gating_output, indices = self.router(x)
        # 2.初始化全零矩阵，后续叠加为最终结果
        # final_output = torch.zeros_like(x)
        final_output = torch.zeros(batch_size, seq_len, self.output_dim).to(x.device)

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        flat_x = x.reshape(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.experts):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1)
            # 5. 展平操作
            flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if flat_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                expert_input = flat_x[flat_mask]
                # 7. 将token输入expert得到输出
                expert_output = expert(expert_input)

                # 8. 计算当前专家对于有作用的token的权重分数
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # 9. 将expert输出乘上权重分数
                weighted_output = expert_output * gating_scores

                # 10. 循环进行做种的结果叠加
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class MoELinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, top_k):
        super(MoELinearLayer, self).__init__()
        self.moe = SparseMoE(in_channels, num_experts, top_k, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()   #[1, 64, 16, 16])
        # 调整维度以适应MoE层：(batch_size, height * width, in_channels)
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, -1, in_channels) # [1, 256, 64]
        # 应用MoE变换
        x_transformed = self.moe(x_flat)
        # 调整回原始形状：(batch_size, out_channels, height, width)
        x_out = x_transformed.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        return x_out


class ChannelRouter(nn.Module):
    def __init__(self, in_channels, num_experts, top_k, out_channels, experts=None,):
        super(ChannelRouter, self).__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.topk = top_k

        # 用一个轻量MLP或者1x1卷积映射通道权重到专家选择概率
        self.routing_layer = nn.Conv2d(in_channels, in_channels * num_experts, kernel_size=1, groups=in_channels)
        self.experts = experts

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 路由得分：[B, C, num_experts]
        raw_scores = self.routing_layer(x).view(B, self.num_experts, C, H, W).mean(dim=[3, 4])  # [B, num_experts, C]
        raw_scores = raw_scores.permute(0, 2, 1)  # [B, C, num_experts]
        if self.topk>0:
            # 只保留 top-k（在 experts 维度上）
            topk_mask = torch.zeros_like(raw_scores)
            topk_idx = torch.topk(raw_scores, self.topk, dim=-1).indices  # [B, C, topk]

            # 用 one-hot 掩码保留 top-k 项
            topk_mask.scatter_(-1, topk_idx, raw_scores.gather(-1, topk_idx))  # 其余位置为0
            final_output = torch.zeros_like(x)

            # 每个专家处理整张特征图
            # expert_outputs = []
            for i, expert in enumerate(self.experts):
                weight = topk_mask[:, :, i].unsqueeze(-1).unsqueeze(-1)

                out = expert(x)  # [B, C, H, W]
                # 路由掩码 [B, C] -> [B, C, 1, 1]
                final_output += out * weight
        else:
            final_output = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                weight = raw_scores[:, :, i].unsqueeze(-1).unsqueeze(-1)
                out = expert(x)  # [B, C, H, W]
                final_output += out * weight
        return final_output  # [B, C, H, W]
    
class GroupedChannelSelection(nn.Module):
    def __init__(self, in_channels, out_channles, groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True)
        )
        self.groups = groups
        channels_per_group = out_channles // groups
        self.group_selection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channles, groups, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        group_weights = self.group_selection(x)  # [B, groups, 1, 1]
        # 扩展到所有通道
        channel_weights = group_weights.repeat_interleave(
            x.size(1) // self.groups, dim=1
        )

        return channel_weights
    

class SpatialNoisyTopkRouter(nn.Module):
    """Spatial top-k router that can operate at *region* (patch) granularity.

    When ``patch_size == 1`` (default) the router is exactly pixel-wise and
    behaves identically to the previous implementation.  When ``patch_size > 1``
    the router map is computed on a down-sampled feature of size
    ``(ceil(H / p), ceil(W / p))``, the top-k assignment is made per patch, and
    the resulting ``indices`` / ``router_output`` maps are nearest-neighbor
    upsampled back to ``(H, W)`` so that all pixels inside a patch share the
    same expert selection.  This is the region-based routing variant used in
    the ablation requested by the reviewer.
    """

    def __init__(self, in_channels, num_experts, top_k, noise_scale=0.2, patch_size=1):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.noise_scale = noise_scale
        self.patch_size = max(int(patch_size), 1)
        self.topkroute_conv = nn.Conv2d(in_channels, num_experts, kernel_size=1)
        self.noise_conv = nn.Conv2d(in_channels, num_experts, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        p = self.patch_size

        if p > 1:
            Hp = max(1, int(math.ceil(H / p)))
            Wp = max(1, int(math.ceil(W / p)))
            x_routing = F.adaptive_avg_pool2d(x, output_size=(Hp, Wp))
        else:
            x_routing = x

        logits = self.topkroute_conv(x_routing)  # [B, E, Hp, Wp]
        noise_logits = self.noise_conv(x_routing)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise * self.noise_scale

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=1)  # [B, topk, Hp, Wp]

        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=1)  # [B, E, Hp, Wp]

        if p > 1:
            router_output = F.interpolate(router_output, size=(H, W), mode='nearest')
            indices = F.interpolate(indices.float(), size=(H, W), mode='nearest').long()

        return router_output, indices

class SpatialExpert(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        return self.net(x)


class SpatialSparseMoE(nn.Module):
    """Sparse MoE with optional region-based (patch-wise) spatial routing.

    The ``patch_size`` argument controls the granularity of expert selection:
    ``patch_size=1`` recovers the original pixel-wise routing, while
    ``patch_size>1`` partitions the spatial map into non-overlapping
    ``patch_size x patch_size`` regions and assigns one expert combination per
    region, reducing the router FLOPs from O(H*W*C*E) to O((H/p)*(W/p)*C*E).
    """

    def __init__(self, n_embed, num_experts, top_k, out_channels, experts=None,
                 noise_scale=0.2, patch_size=1):
        super(SpatialSparseMoE, self).__init__()
        self.patch_size = max(int(patch_size), 1)
        self.router = SpatialNoisyTopkRouter(n_embed, num_experts, top_k,
                                             noise_scale=noise_scale,
                                             patch_size=self.patch_size)
        self.experts = nn.ModuleList([Expert(n_embed, out_channels) for _ in range(num_experts)]) if experts is None else experts
        self.top_k = top_k
        self.num_experts = num_experts
        self.record_routing = None  # None, 'full', or 'lite'
        self.last_routing_stats = None

    def forward(self, x):
        B, C, H, W = x.shape
        gating_output, indices = self.router(x) # [B, E, H, W], [B, topk, H, W]
        final_output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=1) # [B, H, W]
            expert_mask = expert_mask.unsqueeze(1).expand_as(x) # [B, C, H, W]
            
            if expert_mask.any():
                expert_input = torch.where(expert_mask, x, torch.zeros_like(x))
                expert_output = expert(expert_input)  # [B, C, H, W]
                expert_weights = gating_output[:, i:i+1]
                expert_weights = expert_weights.expand(-1, C, -1, -1)
                weighted_output = expert_output * expert_weights
                final_output = final_output + weighted_output

        if self.record_routing:
            with torch.no_grad():
                # For region routing the meaningful unit of a routing decision
                # is one patch, so we down-sample before computing entropy/load
                # statistics to avoid double counting identical neighbours.
                if self.patch_size > 1:
                    Hp = max(1, int(math.ceil(H / self.patch_size)))
                    Wp = max(1, int(math.ceil(W / self.patch_size)))
                    prob = F.adaptive_avg_pool2d(gating_output.detach(),
                                                 output_size=(Hp, Wp))
                    idx_ds = F.adaptive_max_pool2d(indices.float(),
                                                   output_size=(Hp, Wp)).long()
                else:
                    prob = gating_output.detach()
                    idx_ds = indices

                entropy = -(prob * (prob + 1e-8).log()).sum(dim=1).mean()
                load = torch.zeros(self.num_experts, device=x.device)
                for ei in range(self.num_experts):
                    load[ei] = (idx_ds == ei).any(dim=1).float().mean()
                stats = {
                    'entropy': entropy.item(),
                    'load': load.cpu(),
                    'patch_size': self.patch_size,
                }
                if self.record_routing == 'full':
                    stats['indices'] = idx_ds.cpu()
                    stats['gating'] = prob.cpu()
                self.last_routing_stats = stats

        return final_output


class SpatialSparse(nn.Module):
    def __init__(self, in_channels, num_experts, top_k, out_channels, experts=None):
        super().__init__()
        # self.router = SpatialNoisyTopkRouter(in_channels, num_experts, top_k)
        if experts==None:
            self.experts = nn.ModuleList([
            SpatialExpert(in_channels, out_channels) 
            for _ in range(num_experts)
        ])
        else:
            self.experts = experts
        self.top_k = top_k
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.router = nn.Parameter(torch.rand(in_channels, num_experts))
        self.gumble_softmax = GumbelSoftmax()
    
    def forward(self, x):
        # x: [batch_size, in_channels, height, width]
        # print(self.router[:5])
        B, C, H, W = x.shape
        
        # routing_weights, indices = self.router(x)  # [B, C, num_experts]
        # routing_weights = F.softmax(self.router, dim=-1)
        routing_weights = self.gumble_softmax(self.router)
        routing_weights = routing_weights.unsqueeze(0).expand(B, C, self.num_experts)
        
        final_output = torch.zeros(B, self.out_channels, H, W, device=x.device)
        
        for i, expert in enumerate(self.experts):
            expert_mask = routing_weights[:,:,i]
            # 5. 展平操作
            # expand_mask = expert_mask.unsqueeze(-1).unsqueeze(-1).expand(B,C,H,W)

            masked_input = x * expert_mask.unsqueeze(-1).unsqueeze(-1)
                # 7. 将token输入expert得到输出
            expert_output = expert(masked_input)

            final_output = final_output + expert_output
            
        return final_output

class SpatialMoELayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, top_k):
        super().__init__()
        self.moe = SpatialSparseMoE(in_channels, num_experts, top_k, out_channels)
    
    def forward(self, x):
        # x: [batch_size, in_channels, height, width]
        return self.moe(x)



def test_moe_linear_layer():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 2
    in_channels = 64
    height = 32
    width = 32
    out_channels = 128
    num_experts = 4
    top_k = 2
    
    # 创建模型
    model = MoELinearLayer(in_channels, out_channels, num_experts, top_k)
    
    # 创建输入tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 运行前向传播
    output = model(x)
    
    # 测试输出形状
    expected_shape = (batch_size, out_channels, height, width)
    assert output.shape == expected_shape, \
        f"输出形状错误: 期望 {expected_shape}, 得到 {output.shape}"
    
    # 测试输出值是否在合理范围内
    assert not torch.isnan(output).any(), "输出包含 NaN 值"
    assert not torch.isinf(output).any(), "输出包含 Inf 值"
    
    # 测试梯度
    output.sum().backward()
    for param in model.parameters():
        assert param.grad is not None, "某些参数没有梯度"
        assert not torch.isnan(param.grad).any(), "梯度包含 NaN 值"
        assert not torch.isinf(param.grad).any(), "梯度包含 Inf 值"
    
    print("基本功能测试通过")
    
    # 测试不同输入尺寸
    test_sizes = [
        (1, 64, 16, 16),
        (4, 64, 64, 64),
        (8, 64, 8, 8)
    ]
    
    for size in test_sizes:
        x = torch.randn(*size)
        output = model(x)
        expected_shape = (size[0], out_channels, size[2], size[3])
        assert output.shape == expected_shape, \
            f"输入尺寸 {size} 测试失败: 期望输出形状 {expected_shape}, 得到 {output.shape}"
    
    print("不同输入尺寸测试通过")
    
    # 测试设备兼容性
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(batch_size, in_channels, height, width).cuda()
        output = model(x)
        assert output.device.type == "cuda", "GPU 测试失败"
        print("GPU 兼容性测试通过")
    
    # 测试模型行为一致性
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
    
    # 创建模型
    model = SpatialMoELayer(in_channels, out_channels, num_experts, top_k)
    
    # 创建输入
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 前向传播
    output = model(x)
    
    # 检查输出形状
    expected_shape = (batch_size, out_channels, height, width)
    assert output.shape == expected_shape
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("测试通过!")

if __name__ == "__main__":
    test_spatial_moe()
    # try:
    #     test_moe_linear_layer()
    #     print("所有测试通过！")
    # except AssertionError as e:
    #     print(f"测试失败: {str(e)}")
    # except Exception as e:
    #     print(f"发生错误: {str(e)}")