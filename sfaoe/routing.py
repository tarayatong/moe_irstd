"""主路径的空间路由；旧实验路由仍位于 model.mask。"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    def __init__(self, n_embd, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_embd, out_channels, 3, 1, 1), nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class SpatialNoisyTopkRouter(nn.Module):
    """像素／区域级带噪声 Top-K 路由。

    输入为 [B,C,H,W]，返回权重 [B,E,H,W] 与索引 [B,K,H,W]。
    区域模式先做自适应平均池化，再将路由结果用最近邻插值恢复。
    仅训练阶段加入噪声；推理阶段使用确定性 logits。"""

    def __init__(self, in_channels, num_experts, top_k, noise_scale=0.2, patch_size=1):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.noise_scale = noise_scale
        self.patch_size = max(int(patch_size), 1)
        self.topkroute_conv = nn.Conv2d(in_channels, num_experts, kernel_size=1)
        self.noise_conv = nn.Conv2d(in_channels, num_experts, kernel_size=1)

    def forward(self, x):
        (B, _, H, W) = x.shape
        p = self.patch_size
        if p > 1:
            Hp = max(1, int(math.ceil(H / p)))
            Wp = max(1, int(math.ceil(W / p)))
            x_routing = F.adaptive_avg_pool2d(x, output_size=(Hp, Wp))
        else:
            x_routing = x
        # logits: [B,E,Hp,Wp]；Top-K 在专家维上选择，而不是空间维。
        logits = self.topkroute_conv(x_routing)
        if self.training and self.noise_scale > 0:
            noise_logits = self.noise_conv(x_routing)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise * self.noise_scale
        else:
            noisy_logits = logits
        (top_k_logits, indices) = noisy_logits.topk(self.top_k, dim=1)
        # 未选 logits 为 -inf，Softmax 后权重严格为零。
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=1)
        if p > 1:
            router_output = F.interpolate(router_output, size=(H, W), mode="nearest")
            indices = F.interpolate(indices.float(), size=(H, W), mode="nearest").long()
        return (router_output, indices)


class SpatialSparseMoE(nn.Module):
    """空间稀疏专家混合，保留原检查点参数名。

    full：专家读取完整特征，仅输出按 Top-K 权重融合。
    sparse：先按选中位置屏蔽专家输入，再对输出加权。
    两者都会执行稠密专家算子；路由稀疏不等于像素级计算跳过。
    last_aux_loss 保存当前训练前向的负载均衡损失，供训练器收集。
    record_routing 为 None/lite/full，控制可选的统计导出。"""

    def __init__(
        self,
        n_embed,
        num_experts,
        top_k,
        out_channels,
        experts=None,
        noise_scale=0.2,
        patch_size=1,
        input_routing="sparse",
    ):
        super(SpatialSparseMoE, self).__init__()
        ir = str(input_routing).lower()
        if ir not in ("full", "sparse"):
            raise ValueError(
                f"input_routing must be 'full' (V-MoE) or 'sparse' (legacy), got {input_routing!r}"
            )
        self.input_routing = ir
        self.patch_size = max(int(patch_size), 1)
        self.router = SpatialNoisyTopkRouter(
            n_embed,
            num_experts,
            top_k,
            noise_scale=noise_scale,
            patch_size=self.patch_size,
        )
        self.experts = (
            nn.ModuleList([Expert(n_embed, out_channels) for _ in range(num_experts)])
            if experts is None
            else experts
        )
        self.top_k = top_k
        self.num_experts = num_experts
        self.record_routing = None
        self.last_routing_stats = None
        self.last_aux_loss = None

    def _compute_aux_loss(self, gating_output, indices):
        """Switch-style auxiliary load-balancing loss.

        Args:
            gating_output: [B, E, H, W] softmax probabilities (sum to 1 over E).
            indices:       [B, topk, H, W] hard top-k expert assignments.

        At uniform routing this loss equals 1; at full collapse it equals E,
        so adding it with a small weight (e.g. 1e-2) discourages winner-take-
        all behaviour while flowing gradients through ``gating_output``.
        """
        E = self.num_experts
        one_hot = F.one_hot(indices, num_classes=E).float()
        f = one_hot.sum(dim=1).mean(dim=(0, 1, 2))
        P = gating_output.mean(dim=(0, 2, 3))
        aux = E * (f * P).sum() / max(self.top_k, 1)
        return aux

    def _accumulate_experts_vmoe_full_input(self, x, gating_output, final_output):
        """V-MoE: full-map forward per expert, sparse combination via gating only."""
        (_, C, _, _) = x.shape
        for i, expert in enumerate(self.experts):
            expert_weights = gating_output[:, i : i + 1]
            if not expert_weights.any():
                continue
            expert_output = expert(x)
            final_output = final_output + expert_output * expert_weights.expand(
                -1, C, -1, -1
            )
        return final_output

    def _accumulate_experts_legacy_masked_input(
        self, x, gating_output, indices, final_output
    ):
        """Original path: mask the input to each expert by routed pixels (legacy)."""
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=1)
            expert_mask = expert_mask.unsqueeze(1).expand_as(x)
            if expert_mask.any():
                expert_input = torch.where(expert_mask, x, torch.zeros_like(x))
                expert_output = expert(expert_input)
                expert_weights = gating_output[:, i : i + 1].expand(
                    -1, x.size(1), -1, -1
                )
                final_output = final_output + expert_output * expert_weights
        return final_output

    def forward(self, x):
        (B, C, H, W) = x.shape
        (gating_output, indices) = self.router(x)
        # 保留专家累加顺序，使已有权重的数值行为保持一致。
        final_output = torch.zeros_like(x)
        if self.input_routing == "full":
            final_output = self._accumulate_experts_vmoe_full_input(
                x, gating_output, final_output
            )
        else:
            final_output = self._accumulate_experts_legacy_masked_input(
                x, gating_output, indices, final_output
            )
        if self.training:
            self.last_aux_loss = self._compute_aux_loss(gating_output, indices)
        else:
            self.last_aux_loss = None
        if self.record_routing:
            with torch.no_grad():
                if self.patch_size > 1:
                    Hp = max(1, int(math.ceil(H / self.patch_size)))
                    Wp = max(1, int(math.ceil(W / self.patch_size)))
                    prob = F.adaptive_avg_pool2d(
                        gating_output.detach(), output_size=(Hp, Wp)
                    )
                    idx_ds = F.adaptive_max_pool2d(
                        indices.float(), output_size=(Hp, Wp)
                    ).long()
                else:
                    prob = gating_output.detach()
                    idx_ds = indices
                entropy = -(prob * (prob + 1e-08).log()).sum(dim=1).mean()
                load = torch.zeros(self.num_experts, device=x.device)
                for ei in range(self.num_experts):
                    load[ei] = (idx_ds == ei).any(dim=1).float().mean()
                stats = {
                    "entropy": entropy.item(),
                    "load": load.cpu(),
                    "patch_size": self.patch_size,
                }
                if self.record_routing == "full":
                    stats["indices"] = idx_ds.cpu()
                    stats["gating"] = prob.cpu()
                self.last_routing_stats = stats
        return final_output
