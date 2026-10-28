"""SFAoE 训练主路径；请由 train_spar.py 或 python -m sfaoe.training 启动。"""

import os
import random
import time
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.load_param_data import load_dataset, load_dataset_5folders, load_param
from model.loss import spar_iou_loss
from model.metric import ROCMetric, mIoU, P_R_F1
from model.misc import AverageMeter
from model.utils import TrainSetLoader, TestSetLoader, weights_init_xavier, save_model
from sfaoe.factory import build_model
from sfaoe.routing import SpatialSparseMoE


class Trainer(object):
    """单 GPU 训练器：数据加载、优化、评估与路由快照。

    每个实例持有自己的 args，不依赖入口脚本的全局变量。
    保留原损失、指标、检查点选择和调度时机，避免重构改变实验定义。"""

    def __init__(self, args):
        self.args = args
        seed = 40
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.ROC = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PR = P_R_F1(1, 10, self.args.crop_size)
        self.save_prefix = "_".join([self.args.model, self.args.dataset])
        self.save_dir = self.args.save_dir
        (nb_filter, num_blocks) = load_param(self.args.channel_size, self.args.backbone)
        if self.args.mode == "TXT":
            dataset_dir = self.args.root + "/" + self.args.dataset
            (train_img_ids, val_img_ids, test_img_ids) = load_dataset(
                self.args.root, self.args.dataset, self.args.split_method
            )
        if self.args.mode == "SIATD10seq":
            dataset_dir = self.args.root + "/" + self.args.dataset
            (train_img_ids, val_img_ids, test_img_ids) = load_dataset_5folders(
                self.args.root, self.args.dataset, self.args.split_method
            )
        trainset = TrainSetLoader(
            dataset_dir,
            img_id=train_img_ids,
            base_size=self.args.base_size,
            crop_size=self.args.crop_size,
            suffix=self.args.suffix,
            dataset_name=self.args.dataset,
        )
        testset = TestSetLoader(
            dataset_dir,
            img_id=test_img_ids,
            base_size=self.args.base_size,
            crop_size=self.args.crop_size,
            suffix=self.args.suffix,
            dataset_name=self.args.dataset,
        )
        self.train_data = DataLoader(
            dataset=trainset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            drop_last=True,
        )
        self.test_data = DataLoader(
            dataset=testset,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.workers,
            drop_last=False,
        )
        model = build_model(self.args, nb_filter, num_blocks)
        model = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model
        if self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr
            )
        elif self.args.optimizer == "Adagrad":
            self.optimizer = torch.optim.Adagrad(
                filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr
            )
        if self.args.scheduler == "CosineAnnealingLR":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_lr
            )
        # 历史行为：仅此处调用一次。改变调度时机应作为独立实验变更。
        self.scheduler.step()
        self.lossfunc = spar_iou_loss
        self.best_iou = 0
        self.best_f1 = 0
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def model_info(self, model, input):
        """按传入张量统计模型规模；thop 仅在调用此方法时导入。"""
        n_p = sum((x.numel() for x in model.parameters()))
        n_g = sum((x.numel() for x in model.parameters() if x.requires_grad))
        from thop import profile

        (flops, params) = profile(model, inputs=(input,))
        fs = f", {flops / 1000000000.0} GFLOPs"
        print(
            f"model info| summary: {len(list(model.modules()))} layers, {n_p / 1000000.0}M parameters, {n_g / 1000000.0}M gradients{fs}"
        )

    def training(self, epoch):
        """一次训练轮次。深监督主损失加各 MoE 层平均负载均衡损失。"""
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        main_losses = AverageMeter()
        aux_losses = AverageMeter()
        aux_w = float(getattr(self.args, "aux_loss_weight", 0.0))
        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()
            torch.cuda.synchronize()
            start = time.time()
            preds = self.model(data)
            main_loss = self.lossfunc(preds, labels)
            pred = preds[-1]
            # 各 MoE 层取平均，避免辅助损失随网络深度成比例放大。
            aux_loss = self._collect_aux_loss(self.model)
            if aux_w > 0.0 and aux_loss is not None:
                loss = main_loss + aux_w * aux_loss
                aux_val = aux_loss.item()
            else:
                loss = main_loss
                aux_val = 0.0
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            main_losses.update(main_loss.item(), pred.size(0))
            aux_losses.update(aux_val, pred.size(0))
            torch.cuda.synchronize()
            end = time.time()
            infer_time = end - start
            tbar.set_description(
                "Epoch %d | total %.4f  main %.4f  aux %.4f  FPS %.2f"
                % (
                    epoch,
                    losses.avg,
                    main_losses.avg,
                    aux_losses.avg,
                    self.args.train_batch_size / infer_time,
                )
            )
        self.train_loss = losses.avg
        self.train_aux_loss = aux_losses.avg

    def _collect_aux_loss(self, model):
        """Sum the per-module Switch load-balancing loss across all
        SpatialSparseMoE layers and average so the magnitude is roughly
        independent of network depth.  Returns ``None`` when no module has
        produced a loss this step (e.g. all in eval mode)."""
        items = [
            m.last_aux_loss
            for m in model.modules()
            if isinstance(m, SpatialSparseMoE) and m.last_aux_loss is not None
        ]
        if not items:
            return None
        return torch.stack(items).mean()

    def _set_routing_record(self, model, mode):
        for m in model.modules():
            if isinstance(m, SpatialSparseMoE):
                m.record_routing = mode

    def _collect_routing_stats(self, model):
        stats = []
        for name, m in model.named_modules():
            if isinstance(m, SpatialSparseMoE) and m.last_routing_stats is not None:
                stats.append({"name": name, **m.last_routing_stats})
                m.last_routing_stats = None
        return stats

    def save_routing_snapshot(self, epoch, save_dir="./routing_logs", mode="full"):
        """额外执行一次推理导出路由统计，并在异常时恢复模型模式。"""
        os.makedirs(save_dir, exist_ok=True)
        was_training = self.model.training
        self._set_routing_record(self.model, mode)
        try:
            self.model.eval()
            with torch.no_grad():
                (data, labels) = next(iter(self.test_data))
                data = data.cuda()
                self.model(data)
        finally:
            self._set_routing_record(self.model, None)
            self.model.train(was_training)
        all_stats = self._collect_routing_stats(self.model)
        snapshot = {
            "epoch": epoch,
            "modules": all_stats,
            "meta": {
                "noise_scale": getattr(self.args, "noise_scale", None),
                "patch_size": getattr(self.args, "patch_size", None),
                "input_routing": getattr(self.args, "input_routing", None),
                "mode": mode,
            },
        }
        if mode == "full":
            snapshot["input"] = data.cpu()
            snapshot["labels"] = labels.cpu()
        torch.save(snapshot, os.path.join(save_dir, f"routing_epoch_{epoch}.pt"))

    def testing(self, epoch):
        """按原指标实现评估并保存最佳 IoU 检查点；保留既有评估协议。"""
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.PR.reset()
        losses = AverageMeter()
        all_infer_time = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                torch.cuda.synchronize()
                start = time.time()
                preds = self.model(data)
                loss = self.lossfunc(preds, labels)
                pred = preds[-1]
                torch.cuda.synchronize()
                end = time.time()
                infer_time = end - start
                all_infer_time += infer_time
                losses.update(loss.item(), pred.size(0))
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                (ture_positive_rate, false_positive_rate, recall, precision) = (
                    self.ROC.get()
                )
                (_, mean_IOU, _) = self.mIoU.get()
                tbar.set_description(
                    "Epoch %d, test loss %.4f, mean_IoU: %.4f"
                    % (epoch, losses.avg, mean_IOU)
                )
            test_loss = losses.avg
            print(
                "FPS: %.2f"
                % (len(self.test_data) * self.args.test_batch_size / all_infer_time)
            )
        save_model(
            mean_IOU,
            self.best_iou,
            self.save_dir,
            self.save_prefix,
            self.train_loss,
            test_loss,
            recall,
            precision,
            epoch,
            self.model.state_dict(),
        )
        if mean_IOU > self.best_iou:
            self.best_iou = mean_IOU


def main(args):
    """运行原训练流程；路由快照和中间权重由命令行选项控制。"""
    trainer = Trainer(args)
    save_routing = getattr(args, "save_routing", "none")
    routing_epochs = getattr(args, "routing_epochs", set()) or set()
    save_intermediate_ckpt = getattr(args, "save_intermediate_ckpt", False)
    if save_routing != "none" and routing_epochs:
        ps_tag = (
            args.patch_size
            if isinstance(args.patch_size, int)
            else "-".join((str(p) for p in args.patch_size))
        )
        routing_log_dir = os.path.join(
            args.save_dir, f"routing_alpha_{args.noise_scale}_p{ps_tag}"
        )
        os.makedirs(routing_log_dir, exist_ok=True)
        print(
            f"[routing] mode={save_routing}  epochs={sorted(routing_epochs)}  dir={routing_log_dir}"
        )
    else:
        routing_log_dir = None
        print("[routing] disabled (use --save_routing lite|full to enable).")
    ckpt_log_dir = None
    if save_intermediate_ckpt and routing_epochs:
        ckpt_log_dir = os.path.join(args.save_dir, "intermediate_ckpts")
        os.makedirs(ckpt_log_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)
        if epoch in routing_epochs:
            if routing_log_dir is not None:
                trainer.save_routing_snapshot(
                    epoch, save_dir=routing_log_dir, mode=save_routing
                )
            if ckpt_log_dir is not None:
                torch.save(
                    trainer.model.state_dict(),
                    os.path.join(ckpt_log_dir, f"ckpt_epoch_{epoch}.pth"),
                )
    input = torch.randn(1, 3, 256, 256).cuda()
    trainer.model_info(trainer.model, input)


if __name__ == "__main__":
    from sfaoe.config import parse_args

    main(parse_args())
