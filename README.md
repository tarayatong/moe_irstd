# SFAoE: context + spatial + Identity

最终模型分支：`final/context-sparse`。

## 最终结构

- 每个 MoE 固定三个专家，顺序为 **0=context、1=多感受野空间、2=Identity**。
- 主干保留 15 个 MoE、四个分辨率层以及原连接方式。
- 默认 `input_routing=sparse`、TopK=2、noise=0.2、aux=0.01；原膨胀率不变。
- context 沿用原实现：两层 db1 DWT，四个子带分别做深度卷积和可学习缩放，
  由深到浅进行全子带 IWT，输出为输入残差加重建结果。
  **context 不是纯低通或仅保留 LL**，LH/HL/HH 仍参与重建。
- 已移除独立高频专家、DySample 高频增强路径、三子带乘积及新 HFJointWT 试验。
  主干上采样所用的 DySample 保留。
- 移除 context 内从未参与其 forward 的 `llconv` / `hconv` 成员。
  路由导出和图例统一为三个专家，不再沿用四专家标签。

## 代码导航

| 文件 | 职责 |
| --- | --- |
| `sfaoe/experts.py` | context、空间专家、固定三专家组合 |
| `sfaoe/backbone.py` | 多阶段、多分辨率主干 |
| `sfaoe/routing.py` | 原 Top-K 路由、输入掩码和辅助损失 |
| `sfaoe/config.py` | 参数解析；sparse 默认值；TopK 范围 1–3 |
| `sfaoe/factory.py` | 模型构建 |
| `sfaoe/checkpoints.py` | 三专家 context 历史权重的严格兼容清理 |
| `sfaoe/training.py` | 原项目训练、评估和路由快照 |
| `tools/convert_context_checkpoint.py` | 转换已有三专家 context 权重 |
| `tests/` | 模型结构、输入掩码、数值及训练器回归测试 |

`train_spar.py` 是主入口。`model/mobile_mamba.py` 和
`model/model_mask_s_shape.py` 保留转发路径；历史类名不代表使用 Mamba。
其他 `model_*.py`、`train.py`、`train_mask.py`、`test.py` 为既有实验代码，
不作为最终模型的训练/评估入口；其中旧硬编码没有在本次整理中重写。

## 使用

```bash
python train_spar.py --help
python train_spar.py --dataset NUAA-SIRST --input_routing sparse
python -m unittest discover -s tests -v
```

数据目录沿用 `dataset/NUAA-SIRST/images`、`masks`、`50_50/train.txt`、
`50_50/test.txt`；IRSTD 仍按项目原规则读取。依赖 PyTorch、torchvision、
timm、PyWavelets、NumPy、SciPy、scikit-image、Pillow、OpenCV、tqdm、
matplotlib；复杂度统计另需 thop。

此次只收敛模型结构，没有改变训练器的损失、优化器、学习率调度时机或数据处理。
继承的训练默认值仍为 1000 轮和 batch4，显式命令行参数优先；不要把此入口误认为
远程已冻结的 300 轮/batch8/固定学习率实验 runner。
旧训练器仅初始化时调用一次 scheduler.step()，这一历史行为保持原样。
已有实验指标仍属于对应原权重与原运行协议，不能直接标成这次清理后重新训练的结果。

## 历史 checkpoint

已有 `experts_multi_low` 三专家 context 权重可转换；转换仅删除 context 内
从未使用的 `llconv` / `hconv` 键，所有有效张量必须严格匹配。

```bash
python tools/convert_context_checkpoint.py --input /path/to/experts_multi_low/best.pt --output /path/to/context_sparse_best.pt
```

转换不包含优化器状态，输出用于推理/微调，不用于无缝续跑旧训练。
四专家权重的路由维度和专家编号不同，工具会拒绝加载，不静默丢弃 HF 权重。
`visualize_routing.py` 也采用上述严格加载规则；历史四专家导出请使用历史分支。

删除未使用成员会改变从头构建模型时的随机数消耗顺序。因此即使 seed 相同，也不应
声称清理后从头训练会逐位复现原实验。已有 context 权重的 forward 数值保持性单独验证。
数据集、历史权重、实验输出和论文不纳入本分支整理，也不删除。
