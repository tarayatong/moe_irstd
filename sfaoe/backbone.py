"""SFAoE 主干：显式依赖和可独立阅读的节点连接。"""

import logging
import torch
from torch import nn
from torch.nn import functional as F
from model.blocks import DySample
from model.mobilenet_v2_dg_util import ConvBNReLU
from sfaoe.experts import MobileMambaBlock

logging.getLogger("thop").setLevel(logging.WARNING)


class DNANet(nn.Module):
    """SFAoE 多阶段、多分辨率主干。

    历史 API 中 stage 表示分辨率层数，block_count 表示处理阶段数。
    node_list[处理阶段][分辨率层] 的注册顺序保持不变。
    返回深监督 logits 列表，最后一项为多尺度融合后的预测。"""

    def __init__(
        self,
        num_classes,
        input_channels,
        block,
        num_blocks,
        nb_filter,
        stage=4,
        block_count=4,
        moe_stages=None,
        dilations=None,
        noise_scale=0.2,
        patch_size=1,
        top_k=2,
        input_routing="full",
    ):
        """构建网络，patch_size/moe_stages 按分辨率层索引。

        num_blocks 是历史兼容参数；当前 _make_layer 默认仍为两个残差块。
        所有注册成员（含空 ModuleList）保持原名称与创建顺序。"""
        super(DNANet, self).__init__()
        if moe_stages is None:
            moe_stages = [True] * stage
        self.moe_stages = moe_stages
        if patch_size is None:
            patch_size_list = [1] * stage
        elif isinstance(patch_size, (list, tuple)):
            assert len(patch_size) == stage, (
                f"patch_size list length {len(patch_size)} != stage {stage}"
            )
            patch_size_list = [max(int(p), 1) for p in patch_size]
        else:
            patch_size_list = [max(int(patch_size), 1)] * stage
        self.patch_size_list = patch_size_list
        self.top_k = max(int(top_k), 1)
        ir = str(input_routing).lower()
        assert ir in ("full", "sparse"), (
            f"input_routing must be 'full' or 'sparse', got {input_routing!r}"
        )
        self.input_routing = ir
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.stage = stage
        self.block = block_count
        self.node_list = nn.ModuleList(
            [
                nn.ModuleList([nn.ModuleList([]) for _ in range(stage)])
                for _ in range(block_count)
            ]
        )
        self.channel_squeeze_list = nn.ModuleList(
            [
                nn.ModuleList([nn.ModuleList([]) for _ in range(stage)])
                for _ in range(block_count)
            ]
        )
        self.up_list = nn.ModuleList(
            [DySample(nb_filter[i], scale=2**i) for i in range(stage)]
        )
        self.nb_filter = nb_filter
        self.connect_type = "cross"
        assert self.connect_type in ["cross", "fore", "dense"]
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        for j in range(self.block):
            for i in range(self.stage):
                if i + j == 0:
                    continue
                if self.connect_type == "fore":
                    if j != 2:
                        inp_c = nb_filter[0] * (j * (2**self.stage - 1) + (2**i - 1))
                    else:
                        inp_c = nb_filter[0] * (
                            j * (2**self.stage - 1)
                            + 2 ** (i + 1) * (2 ** (self.stage - 1 - i) - 1)
                        )
                elif self.connect_type == "cross":
                    if j != 2:
                        inp_c = nb_filter[i] * j + nb_filter[0] * (2**i - 1)
                    else:
                        inp_c = nb_filter[i] * j + nb_filter[0] * 2 ** (i + 1) * (
                            2 ** (self.stage - 1 - i) - 1
                        )
                elif self.connect_type == "dense":
                    inp_c = (
                        nb_filter[i] * j
                        + nb_filter[0] * (2**i - 1)
                        + (
                            nb_filter[0] * (int(2 ** (i - 1) * 7) if i < 3 else 12)
                            - nb_filter[i]
                        )
                        * min(j, 1)
                    )
                if self.moe_stages[i]:
                    self.node_list[j][i] = nn.Sequential(
                        ConvBNReLU(inp_c, nb_filter[i], 3),
                        MobileMambaBlock(
                            "s",
                            nb_filter[i],
                            0.7,
                            0.2,
                            5,
                            0,
                            ssm_ratio=2,
                            layer=i,
                            dilations=dilations,
                            noise_scale=noise_scale,
                            patch_size=self.patch_size_list[i],
                            top_k=self.top_k,
                            input_routing=self.input_routing,
                        ),
                    )
                else:
                    self.node_list[j][i] = self._make_layer(
                        block, inp_c, nb_filter[i], stride=1
                    )
        self.conv1x1 = nn.ModuleList([nn.ModuleList([]) for _ in range(stage)])
        for i in range(self.stage):
            self.conv1x1[i] = nn.Sequential(
                nn.Conv2d(nb_filter[i], nb_filter[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU(inplace=True),
            )
        self.conv0_4_final = self._make_layer(
            block, nb_filter[0] * self.stage, nb_filter[0], 1
        )
        self.final_list = nn.ModuleList([nn.ModuleList([]) for _ in range(block_count)])
        for i in range(self.block):
            self.final_list[i] = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(
        self, block, input_channels, output_channels, num_blocks=2, stride=1
    ):
        layers = []
        layers.append(block(input_channels, output_channels, stride=stride))
        for i in range(num_blocks - 1):
            layers.extend([block(output_channels, output_channels, stride=1)])
        return nn.Sequential(*layers)

    def arch_parameters(self):
        return [param for (name, param) in self.named_parameters() if "arch" in name]

    def weight_parameters(self):
        return [
            param for (name, param) in self.named_parameters() if "arch" not in name
        ]

    def weight_concat(self, feats, weight):
        """
        feat: bcwh
        feats: feat list
        weight:
        """
        w = torch.sigmoid(weight)
        new = torch.concat([feat * w[i] for (i, feat) in enumerate(feats)], dim=1)
        return new

    def forward(self, input):
        """前两遍与第四遍由细到粗，第三遍由粗到细；最后对齐并融合各层特征。"""
        features = [[[] for _ in range(self.stage)] for _ in range(self.block)]
        features[0][0] = self.conv0_0(input)
        concatenate = {
            "cross": self._cat_input_cross,
            "fore": self._cat_input_fore,
            "dense": self._cat_input_dense,
        }[self.connect_type]
        # 第三遍反向遍历分辨率；其余遍次沿高分辨率到低分辨率。
        for pass_index in range(self.block):
            levels = (
                range(self.stage - 1, -1, -1) if pass_index == 2 else range(self.stage)
            )
            for level in levels:
                if pass_index == 0 and level == 0:
                    continue
                node_input = concatenate(features, level, pass_index)
                features[pass_index][level] = self.node_list[pass_index][level](
                    node_input
                )
        # 低分辨率特征先动态上采样并投影到同一通道数，再融合。
        aligned_features = [
            self.conv1x1[level](self.up_list[level](features[-1][level]))
            if level > 0
            else features[-1][level]
            for level in range(self.stage)
        ]
        fused = self.conv0_4_final(torch.concat(aligned_features, 1))
        # 深监督输出保持原列表次序，训练损失和评估均依赖最后一项。
        outputs = [
            self.final_list[index](features[index][0])
            for index in range(self.block - 1)
        ]
        outputs.append(self.final_list[-1](fused))
        return outputs

    def _cat_input_cross(self, feat_list, stage, block):
        """对齐空间尺寸并按原通道顺序拼接已计算的节点；拼接顺序影响旧权重。"""
        if block > 0:
            (b, c, w, h) = feat_list[block - 1][stage].size()
        else:
            (b, c, w, h) = feat_list[block][stage - 1].size()
            w = w // 2
            h = h // 2
        in_j = (
            torch.cat([feat_list[j][stage] for j in range(block)], 1)
            if block > 0
            else None
        )
        if block != 2:
            in_i = (
                torch.cat(
                    [
                        F.interpolate(feat_list[block][i], (w, h), mode="bilinear")
                        for i in range(stage)
                    ],
                    1,
                )
                if stage > 0
                else None
            )
        else:
            in_i = (
                torch.cat(
                    [
                        F.interpolate(feat_list[block][i], (w, h), mode="bilinear")
                        for i in range(self.stage - 1, stage, -1)
                    ],
                    1,
                )
                if stage < self.stage - 1
                else None
            )
        if in_i is None and in_j is None:
            return None
        elif in_i is None:
            return in_j
        elif in_j is None:
            return in_i
        else:
            return torch.cat([in_j, in_i], dim=1)

    def _cat_input_dense(self, feat_list, stage, block):
        """对齐空间尺寸并按原通道顺序拼接已计算的节点；拼接顺序影响旧权重。"""
        if block > 0:
            (b, c, w, h) = feat_list[block - 1][stage].size()
        else:
            (b, c, w, h) = feat_list[block][stage - 1].size()
            w = w // 2
            h = h // 2
        in_j = (
            torch.cat([feat_list[j][stage] for j in range(block - 1)], 1)
            if block > 1
            else None
        )
        in_i = (
            torch.cat(
                [
                    F.interpolate(feat_list[block][i], (w, h), mode="bilinear")
                    for i in range(stage)
                ],
                1,
            )
            if stage > 0
            else None
        )
        in_dense = (
            torch.cat(
                [
                    F.interpolate(feat_list[block - 1][i], (w, h), mode="bilinear")
                    for i in range(max(stage - 1, 0), min(self.stage, stage + 2))
                ],
                1,
            )
            if block > 0
            else None
        )
        valid_tensors = []
        if in_j is not None:
            valid_tensors.append(in_j)
        if in_i is not None:
            valid_tensors.append(in_i)
        if in_dense is not None:
            valid_tensors.append(in_dense)
        if valid_tensors:
            return torch.cat(valid_tensors, 1)
        else:
            return None

    def _cat_input_fore(self, feat_list, stage, block):
        """对齐空间尺寸并按原通道顺序拼接已计算的节点；拼接顺序影响旧权重。"""
        if block > 0:
            (b, c, w, h) = feat_list[block - 1][stage].size()
        else:
            (b, c, w, h) = feat_list[block][stage - 1].size()
            w = w // 2
            h = h // 2
        if block != 2:
            in_i = (
                torch.cat(
                    [
                        F.interpolate(feat_list[block][i], (w, h), mode="bilinear")
                        for i in range(stage)
                    ],
                    1,
                )
                if stage > 0
                else None
            )
        else:
            in_i = (
                torch.cat(
                    [
                        F.interpolate(feat_list[block][i], (w, h), mode="bilinear")
                        for i in range(self.stage - 1, stage, -1)
                    ],
                    1,
                )
                if stage < self.stage - 1
                else None
            )
        in_fore = (
            torch.cat(
                [
                    F.interpolate(feat_list[j][i], (w, h), mode="bilinear")
                    for i in range(self.stage)
                    for j in range(block)
                ],
                1,
            )
            if block > 0
            else None
        )
        if in_i is None and in_fore is None:
            return None
        elif in_i is None:
            return in_fore
        elif in_fore is None:
            return in_i
        else:
            return torch.cat([in_fore, in_i], dim=1)
