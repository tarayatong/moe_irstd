# SFAoE

红外小目标检测项目。论文主路径为 `train_spar.py` → `sfaoe.training` → `sfaoe.backbone`。

## 代码导航

| 文件 | 职责 |
| --- | --- |
| `sfaoe/config.py` | 命令行、默认配置、类型规范化与运行目录初始化 |
| `sfaoe/factory.py` | 模型构建，按需导入实验分支 |
| `sfaoe/backbone.py` | 多阶段、多分辨率主干及特征连接 |
| `sfaoe/experts.py` | 多分支空间专家、小波专家和残差混合块 |
| `sfaoe/routing.py` | 像素/区域 Top-K 路由、负载均衡损失和路由统计 |
| `sfaoe/training.py` | 数据加载、训练、评估、检查点及路由快照 |
| `model/utils.py` | 既有数据集加载、预处理与结果保存 |
| `model/metric.py`, `model/loss.py` | 既有指标和深监督损失定义 |
| `tests/` | 参数优先级、路由与兼容性测试 |

`model/model_mask_s_shape.py`、`model/mobile_mamba.py`、`model/parse_args_train.py` 和
`train_spar.py` 保留旧入口并转发到新模块。`model.mask` 保留其他实验路由，主空间路由
转发到同一个新类，因此既有 `isinstance` 检查仍有效。

## 环境与数据

使用已有 PyTorch 环境。主训练路径依赖 PyTorch、torchvision、NumPy、timm、
PyWavelets、OpenCV、Pillow、SciPy、scikit-image、matplotlib、tqdm；结束时的
模型复杂度统计还需 thop。当前主训练器使用单张 CUDA GPU。

按已有数据目录命名使用 `NUAA-SIRST`；论文中的 SIRST 不要求重命名磁盘数据集。

```text
dataset/NUAA-SIRST/
  images/
  masks/
  50_50/train.txt
  50_50/test.txt
```

其他数据集的目录规则仍由 `model/utils.py` 和 `model/load_param_data.py` 决定。

## 运行

```bash
python train_spar.py --help
python train_spar.py --dataset NUAA-SIRST --epochs 1000 --input_routing sparse
python train_spar.py --dataset IRSTD --base_size 512 --crop_size 512 --input_routing full --save_routing none --routing_epochs ""
python train_spar.py --patch_size 1,2,4,8 --moe_stages 1,1,1,1
python -m unittest discover -s tests -v
```

所有命令行显式值现在都会保留，不再在解析后被示例配置覆盖。无参数运行保留原脚本的
有效默认配置，包括 1000 epochs、sparse 输入及 lite 路由记录。
`--save_dir name` 指定新的运行名，检查点保存到 `result/name/`。
运行名已存在时报错，避免混写结果。路由快照仍沿用原脚本的 `name/routing_.../`
路径约定，未将既有导出目录迁移。

## 阅读模型

`stage` 是历史变量名，代表**分辨率层**；`block_count` 代表**处理阶段**。
主干先由细到粗处理两遍，第三遍由粗到细，第四遍由细到粗。
每个节点按原有顺序拼接已计算的特征；输出是深监督 logits 列表，最后一项是最终预测。

默认专家顺序为高频、低频、空间和 Identity。两种小波专家都保留全部子带；高频专家
增加显式细节增强路径。`input_routing=full` 为完整输入后稀疏输出融合，`sparse` 为
输入先屏蔽再融合。稀疏权重不代表稠密专家算子跳过了所有未路由像素的计算。

参数名、子模块注册顺序、历史 `spacial_moe` 拼写及旧类名保持兼容；推荐保存/加载
`state_dict`。`num_blocks` 仍是主干的历史兼容参数，局部 `_make_layer` 默认使用两个块。

## 兼容边界

`train.py`、`train_mask.py`、`test.py` 及其他 `model_*.py` 是保留的实验入口/变体，
本次没有把不同算法强行合并，也未删除数据、权重或结果。`test.py` 原有硬编码与加载流程
尚未统一，不能用它的命令行选项推断新训练入口的行为。

本次保留原损失、指标实现、测试集选优方式和学习率调度调用位置。
特别注意：旧训练器只在初始化时调用一次 `scheduler.step()`，没有每轮调用；
这会影响实际学习率计划，需要在后续实验协议变更中单独处理。
`ReduceLROnPlateau`、compile、深监督开关等旧选项不代表主训练器已实现相应行为。
这次仅修正参数覆盖、布尔解析、实例参数依赖和快照异常恢复，不宣称复现了完整训练结果。

重构前的完整源码备份、逐项变更和张量等价验证记录位于论文工作目录
`work/refactor_20261026/`。
