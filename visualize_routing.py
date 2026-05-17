"""
Routing visualization for SpatialSparseMoE ablation study on noise scale alpha.

Usage:
    # 1. Plot routing dynamics across training epochs for a single alpha
    python visualize_routing.py --mode dynamics --routing_dir result/routing_alpha_0.2

    # 2. Compare routing dynamics across multiple alpha values
    python visualize_routing.py --mode compare --routing_dirs result/routing_alpha_0.0 result/routing_alpha_0.1 result/routing_alpha_0.2 result/routing_alpha_0.5

    # 2b. Same as (2), but each item is an experiment root that contains exactly one routing_alpha_* folder
    python visualize_routing.py --mode compare --run_roots NUAA-SIRST_DNANet_... NUAA-SIRST_DNANet_...

    # 3. Expert selection fraction vs epoch for one run (e.g. α=0.1)
    python visualize_routing.py --mode expert_loads --routing_dir result/routing_alpha_0.1

    # 3b. Same curves for every α (subplot grid), via experiment roots or explicit routing dirs
    python visualize_routing.py --mode expert_loads_grid --run_roots NUAA-SIRST_DNANet_...

    # 4. Visualize per-pixel expert assignment on test images
    python visualize_routing.py --mode heatmap --routing_dir result/routing_alpha_0.2 --epoch 999

    # 5. Export test images + full per-layer router tensors (gating / indices) from a checkpoint
    #    Each run saves --num_images samples; use --start_batch to continue (next window of batches).
    python visualize_routing.py --mode export --checkpoint path/to.pth.tar --out_dir routing_export/run0 \\
        --dataset NUAA-SIRST --root dataset/ --num_images 10 --start_batch 0 --test_batch_size 1
    python visualize_routing.py --mode export --checkpoint path/to.pth.tar --out_dir routing_export/run1 \\
        --num_images 10 --start_batch 10 --test_batch_size 1

    # 5b. IRSTD 默认导出参数见仓库根目录 export_routing_irstd_default.sh（可通过环境变量覆盖）
    # 6. 将已有 routing.pt 转为 routing_vis 下的 PNG：--mode routing_png --routing_export_dir routing_export/IRSTD
"""

import argparse
import json
import os
import re
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image

from torch.utils.data import DataLoader

from model.load_param_data import load_dataset, load_dataset_5folders, load_param
from model.mask import SpatialSparseMoE
from model.model_DNANet import Res_CBAM_block
from model.utils import TestSetLoader


def load_snapshot(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_all_snapshots(routing_dir):
    files = sorted(glob.glob(os.path.join(routing_dir, 'routing_epoch_*.pt')),
                   key=lambda f: int(f.split('_')[-1].replace('.pt', '')))
    snapshots = []
    for f in files:
        snapshots.append(load_snapshot(f))
    return snapshots


def routing_alpha_from_path(routing_dir):
    """Parse α from .../routing_alpha_<float>."""
    base = os.path.basename(routing_dir.rstrip('/'))
    if not base.startswith('routing_alpha_'):
        return None
    return float(base.split('_')[-1])


def discover_routing_dirs_from_run_roots(run_roots):
    """
    Each run root is expected to contain one routing_alpha_* directory (typical layout per ablation run).
    Returns sorted list of absolute routing log paths (by α ascending).
    """
    discovered = []
    for root in run_roots:
        root = os.path.abspath(os.path.expanduser(root))
        matches = sorted(glob.glob(os.path.join(root, 'routing_alpha_*')))
        if len(matches) == 1:
            discovered.append(matches[0])
        elif len(matches) == 0:
            if os.path.isdir(root) and os.path.basename(root).startswith('routing_alpha_'):
                discovered.append(root)
            else:
                print(f"Warning: no routing_alpha_* under {root}")
        else:
            print(f"Warning: multiple routing_alpha_* under {root}, using all ({len(matches)})")
            discovered.extend(matches)
    discovered = sorted(set(discovered), key=lambda p: routing_alpha_from_path(p) or 0.0)
    return discovered


def expert_display_names(num_experts):
    """Expert index → display name (0 高频, 1 低频, 2 local, 3 identity)."""
    known = ['高频专家', '低频专家', 'local专家', 'identity']
    names = [known[i] if i < len(known) else f'Expert {i}' for i in range(num_experts)]
    return names


def _parse_patch_size_arg(patch_size_str):
    if patch_size_str is None or str(patch_size_str).strip() == '':
        return 1
    s = str(patch_size_str).strip()
    if ',' in s:
        return [int(x) for x in s.split(',') if x.strip() != '']
    return int(s)


def _denormalize_vis(tensor_chw, dataset_name):
    """Invert TestSetLoader Normalize → RGB uint8 [H,W,3]. tensor_chw: [3,H,W] float."""
    cfg = {
        'NUAA-SIRST': (101.06385040283203 / 255.0, 34.619606018066406 / 255.0),
        'NUDT-SIRST': (107.80905151367188 / 255.0, 33.02274703979492 / 255.0),
        'IRSTD': (87.4661865234375 / 255.0, 39.719612693969726 / 255.0),
        'SIATD_10seq': (101.06385040283203 / 255.0, 34.619606018066406 / 255.0),
        'DSAT': (101.06385040283203 / 255.0, 34.619606018066406 / 255.0),
    }
    if dataset_name not in cfg:
        mean, std = 0.485, 0.229
    else:
        mean, std = cfg[dataset_name]
    t = tensor_chw.detach().cpu().float()
    m = torch.tensor([mean] * 3).view(3, 1, 1)
    s = torch.tensor([std] * 3).view(3, 1, 1)
    rgb = (t * s + m).clamp(0.0, 1.0).numpy()
    return (rgb * 255.0).round().astype(np.uint8).transpose(1, 2, 0)


def _safe_filename(name):
    name = str(name).replace('/', '_').replace('\\', '_').replace(' ', '_')
    name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)
    return name[:180] if len(name) > 180 else name


def _upsample_map_2d(arr, patch_size, th, tw):
    """Resize a spatial routing map to the requested image size with nearest sampling."""
    arr = np.asarray(arr, dtype=np.float64)
    th, tw = int(th), int(tw)
    if arr.shape == (th, tw):
        return arr

    t = torch.from_numpy(arr).float().view(1, 1, arr.shape[0], arr.shape[1])
    out = torch.nn.functional.interpolate(t, size=(th, tw), mode='nearest')
    return out.view(th, tw).numpy().astype(np.float64)


def _tensor_hw_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _save_routing_pngs(sample_dir, modules_out, image_hw):
    """
    Save routing as viewable PNGs under sample_dir/routing_vis/:
    per layer — expert_map (RGB tab10), entropy (viridis), gating_e{k}.png (grayscale prob).
    modules_out: list of dicts with gating [E,H,W], indices [topk,H,W], routing_entropy_map [H,W], patch_size, name.
    """
    if not modules_out:
        return
    th, tw = int(image_hw[0]), int(image_hw[1])
    vis_dir = os.path.join(sample_dir, 'routing_vis')
    os.makedirs(vis_dir, exist_ok=True)

    for mi, mod in enumerate(modules_out):
        name = mod['name']
        short = name.split('.')[-2] if '.' in name else name
        short = _safe_filename(short)[:48]
        prefix = f'm{mi:02d}_{short}'
        ps = int(mod.get('patch_size', 1))

        gating = _tensor_hw_to_numpy(mod['gating'])
        indices_t = mod['indices']
        if torch.is_tensor(indices_t):
            indices_t = indices_t.detach().cpu().long()
        else:
            indices_t = torch.from_numpy(np.asarray(indices_t)).long()
        dominant = indices_t[0].numpy().astype(np.float64)

        ent = mod['routing_entropy_map']
        ent = _tensor_hw_to_numpy(ent)

        dom_u = _upsample_map_2d(dominant, ps, th, tw)
        E = gating.shape[0]
        vmax = max(E - 1, 1)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap = plt.cm.get_cmap('tab10', max(E, 3))
        rgba = cmap(norm(dom_u))
        rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(vis_dir, f'{prefix}_expert_map.png'))

        ent_u = _upsample_map_2d(ent.astype(np.float64), ps, th, tw)
        lo, hi = float(ent_u.min()), float(ent_u.max())
        en = (ent_u - lo) / (hi - lo + 1e-8)
        cmap_v = plt.cm.get_cmap('viridis')
        rgba_e = cmap_v(en)
        rgb_e = (np.clip(rgba_e[:, :, :3], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_e).save(os.path.join(vis_dir, f'{prefix}_entropy.png'))

        for ei in range(E):
            g_u = _upsample_map_2d(gating[ei], ps, th, tw)
            g_uint8 = (np.clip(g_u, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(g_uint8, mode='L').save(
                os.path.join(vis_dir, f'{prefix}_gating_e{ei}.png'))


def routing_pt_to_pngs(routing_pt_path):
    """Load one routing.pt (export format) and write routing_vis/*.png next to it."""
    routing_pt_path = os.path.abspath(routing_pt_path)
    d = load_snapshot(routing_pt_path)
    sample_dir = os.path.dirname(routing_pt_path)
    mods = d['modules']
    if not mods:
        print(f'No modules in {routing_pt_path}')
        return
    ps = int(mods[0]['patch_size'])
    g0 = mods[0]['gating']
    if torch.is_tensor(g0):
        _, H, W = g0.shape
    else:
        _, H, W = g0.shape
    if ps <= 1:
        th, tw = H, W
    else:
        th, tw = H * ps, W * ps
    _save_routing_pngs(sample_dir, mods, (th, tw))
    print(f'Wrote routing_vis/*.png under {sample_dir}')


def routing_export_dir_to_pngs(export_dir):
    """Find every .../routing.pt under export_dir and convert to PNG."""
    export_dir = os.path.abspath(export_dir)
    pattern = os.path.join(export_dir, '**', 'routing.pt')
    pts = sorted(glob.glob(pattern, recursive=True))
    if not pts:
        print(f'No routing.pt found under {export_dir}')
        return
    for pt in pts:
        routing_pt_to_pngs(pt)


def _set_routing_record(model, mode):
    for m in model.modules():
        if isinstance(m, SpatialSparseMoE):
            m.record_routing = mode


def _collect_routing_stats(model):
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, SpatialSparseMoE) and m.last_routing_stats is not None:
            d = dict(m.last_routing_stats)
            stats.append({
                'name': name,
                'entropy': d.get('entropy'),
                'load': d.get('load'),
                'patch_size': int(d.get('patch_size', 1)),
                'gating': d.get('gating'),
                'indices': d.get('indices'),
            })
            m.last_routing_stats = None
    return stats


def export_test_routing(
        checkpoint_path,
        out_dir,
        *,
        start_batch=0,
        num_images=10,
        test_batch_size=1,
        dataset='NUAA-SIRST',
        root='dataset/',
        split_method='50_50',
        suffix='.png',
        data_mode='TXT',
        workers=0,
        base_size=256,
        crop_size=256,
        channel_size='three',
        backbone='resnet_18',
        moe_stages_str='1,1,1,1',
        dilations_str='1,2,2,3',
        noise_scale=0.2,
        patch_size_str='1',
        top_k=2,
        input_routing='full',
        in_channels=3,
        device='auto',
):
    """Run the model on consecutive test-loader batches and dump inputs, labels, and full router tensors.

    Uses ``model.utils.TestSetLoader`` only (same family as ``train_spar.Trainer`` test DataLoader): one
    deterministic resize to ``base_size``, then ToTensor + dataset-specific Normalize. No random scale,
    crop, flip, color jitter, blur, or padding — those exist only on ``TrainSetLoader`` (training).
    """
    from model.model_mask_s_shape import DNANet

    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    os.makedirs(out_dir, exist_ok=True)

    if data_mode == 'TXT':
        _, _, test_img_ids = load_dataset(root, dataset, split_method)
    elif data_mode == 'SIATD10seq':
        _, _, test_img_ids = load_dataset_5folders(root, dataset, split_method)
    else:
        raise ValueError(f"Unknown data_mode {data_mode!r}, use TXT or SIATD10seq")

    dataset_dir = os.path.join(root, dataset)
    testset = TestSetLoader(
        dataset_dir,
        img_id=test_img_ids,
        base_size=base_size,
        crop_size=crop_size,
        suffix=suffix,
        dataset_name=dataset,
    )
    loader = DataLoader(
        dataset=testset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )

    moe_stages = [bool(int(x)) for x in moe_stages_str.split(',')]
    dilations = [int(x) for x in dilations_str.split(',')]
    patch_size = _parse_patch_size_arg(patch_size_str)

    nb_filter, num_blocks = load_param(channel_size, backbone)
    model = DNANet(
        num_classes=1,
        input_channels=in_channels,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        moe_stages=moe_stages,
        dilations=dilations,
        noise_scale=noise_scale,
        patch_size=patch_size,
        top_k=top_k,
        input_routing=input_routing,
    )
    model = model.to(dev)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'Warning: load_state_dict missing keys ({len(missing)}):', missing[:8], '...' if len(missing) > 8 else '')
    if unexpected:
        print(f'Warning: load_state_dict unexpected keys ({len(unexpected)}):', unexpected[:8], '...' if len(unexpected) > 8 else '')

    saved = []
    saved_count = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            if batch_idx < start_batch:
                continue
            if saved_count >= num_images:
                break

            data = data.to(dev)
            labels = labels.to(dev)

            _set_routing_record(model, 'full')
            model(data)
            _set_routing_record(model, None)

            batch_stats = _collect_routing_stats(model)
            if not batch_stats or batch_stats[0].get('gating') is None:
                raise RuntimeError(
                    'No routing tensors captured. Ensure checkpoint matches a MoE DNANet and SpatialSparseMoE ran with record_routing.')

            B = data.shape[0]
            for b in range(B):
                if saved_count >= num_images:
                    break

                ds_idx = batch_idx * test_batch_size + b
                if ds_idx >= len(test_img_ids):
                    print(f'Warning: dataset index {ds_idx} beyond test list length {len(test_img_ids)}')
                    break

                img_name = test_img_ids[ds_idx]
                safe = _safe_filename(img_name)
                sub = os.path.join(out_dir, f'ds{ds_idx:05d}_{safe}')
                os.makedirs(sub, exist_ok=True)

                rgb = _denormalize_vis(data[b], dataset)
                Image.fromarray(rgb).save(os.path.join(sub, 'input_rgb.png'))

                lbl = labels[b, 0].detach().cpu().float().clamp(0, 1).numpy()
                Image.fromarray((lbl * 255).astype(np.uint8)).save(os.path.join(sub, 'label.png'))

                modules_out = []
                for mod in batch_stats:
                    gating_b = mod['gating'][b].cpu().contiguous()
                    indices_b = mod['indices'][b].cpu().contiguous()
                    entropy_map = -(gating_b * (gating_b + 1e-8).log()).sum(dim=0)
                    modules_out.append({
                        'name': mod['name'],
                        'patch_size': mod['patch_size'],
                        'entropy_scalar_batch': mod['entropy'],
                        'load_batch': mod['load'],
                        'gating': gating_b.float(),
                        'indices': indices_b.long(),
                        'routing_entropy_map': entropy_map.float(),
                    })

                torch.save({
                    'dataset_index': ds_idx,
                    'batch_index': batch_idx,
                    'slot_in_batch': b,
                    'img_name': img_name,
                    'test_batch_size': test_batch_size,
                    'modules': modules_out,
                }, os.path.join(sub, 'routing.pt'))

                _save_routing_pngs(sub, modules_out, (int(data.shape[2]), int(data.shape[3])))

                meta = {
                    'dataset_index': ds_idx,
                    'batch_index': batch_idx,
                    'slot_in_batch': b,
                    'img_name': img_name,
                    'folder': os.path.basename(sub),
                }
                saved.append(meta)
                saved_count += 1

    summary_path = os.path.join(out_dir, 'export_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': os.path.abspath(checkpoint_path),
            'start_batch': start_batch,
            'num_images_requested': num_images,
            'num_images_saved': len(saved),
            'test_batch_size': test_batch_size,
            'dataset': dataset,
            'samples': saved,
        }, f, indent=2, ensure_ascii=False)

    print(f'Exported {len(saved)} samples under {os.path.abspath(out_dir)}')
    print(f'Summary: {summary_path}')
    if saved_count == 0:
        print('No samples saved — check start_batch or num_images vs loader length.')
    return saved


# ========== Figure 1: Routing Dynamics (entropy + load balance over epochs) ==========

def plot_dynamics(routing_dir, save_path=None):
    snapshots = load_all_snapshots(routing_dir)
    if not snapshots:
        print(f"No snapshots found in {routing_dir}")
        return

    epochs = [s['epoch'] for s in snapshots]
    num_modules = len(snapshots[0]['modules'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for mi in range(num_modules):
        entropies = [s['modules'][mi]['entropy'] for s in snapshots]
        axes[0].plot(epochs, entropies, marker='o', markersize=3,
                     label=f"Module {mi}: {snapshots[0]['modules'][mi]['name'].split('.')[-2]}")

    axes[0].set_ylabel('Router Entropy')
    axes[0].set_title('Router Entropy over Training')
    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    num_experts = len(snapshots[0]['modules'][0]['load'])
    expert_names = expert_display_names(num_experts)
    for ei in range(num_experts):
        loads = []
        for s in snapshots:
            avg_load = np.mean([s['modules'][mi]['load'][ei].item() for mi in range(num_modules)])
            loads.append(avg_load)
        axes[1].plot(epochs, loads, marker='s', markersize=3, label=expert_names[ei])

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Expert Load (fraction of pixels)')
    axes[1].set_title('Expert Load Balance over Training')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ========== Expert load only (one α): four experts’ pixel fraction vs epoch ==========

def expert_load_curves_from_snapshots(snapshots):
    """
    Returns (epochs, load_matrix) where load_matrix has shape (n_epochs, n_experts);
    each value is mean over MoE modules of that expert's pixel fraction (same as plot_dynamics).
    """
    epochs = [s['epoch'] for s in snapshots]
    num_modules = len(snapshots[0]['modules'])
    num_experts = len(snapshots[0]['modules'][0]['load'])
    load_matrix = np.zeros((len(snapshots), num_experts), dtype=np.float64)
    for si, s in enumerate(snapshots):
        for ei in range(num_experts):
            load_matrix[si, ei] = np.mean([s['modules'][mi]['load'][ei].item() for mi in range(num_modules)])
    return epochs, load_matrix


def plot_expert_loads(routing_dir, save_path=None):
    """
    For each expert k, plot mean over MoE modules of load[k] (fraction of routed pixels),
    same aggregation as the lower panel of plot_dynamics.
    """
    snapshots = load_all_snapshots(routing_dir)
    if not snapshots:
        print(f"No snapshots found in {routing_dir}")
        return

    alpha = routing_alpha_from_path(routing_dir)
    alpha_tag = f'{alpha:g}' if alpha is not None else routing_dir.rstrip('/').split('_')[-1]

    epochs, load_matrix = expert_load_curves_from_snapshots(snapshots)
    num_experts = load_matrix.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    expert_names = expert_display_names(num_experts)
    for ei in range(num_experts):
        ax.plot(epochs, load_matrix[:, ei], marker='s', markersize=4, linewidth=1.8, label=expert_names[ei])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Expert load (pixel fraction, mean over MoE layers)')
    ax.set_title(f'Expert selection vs epoch (α={alpha_tag})')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_expert_loads_grid(routing_dirs, save_path=None, ncols=4):
    """One subplot per α: four expert load curves vs epoch (sorted by α)."""
    def _sort_key(d):
        a = routing_alpha_from_path(d)
        return (a is not None, a if a is not None else 0.0, d)

    routing_dirs = sorted(routing_dirs, key=_sort_key)
    n = len(routing_dirs)
    if n == 0:
        print('No routing directories to plot')
        return

    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.2 * nrows), sharex=False, sharey=True)
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.atleast_2d(axes)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    first_ax_for_legend = None
    for idx, routing_dir in enumerate(routing_dirs):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        alpha = routing_alpha_from_path(routing_dir)
        alpha_tag = f'{alpha:g}' if alpha is not None else routing_dir.rstrip('/').split('_')[-1]

        snapshots = load_all_snapshots(routing_dir)
        if not snapshots:
            ax.set_visible(False)
            print(f"Warning: no snapshots in {routing_dir}, skipped")
            continue

        epochs, load_matrix = expert_load_curves_from_snapshots(snapshots)
        num_experts = load_matrix.shape[1]
        expert_names = expert_display_names(num_experts)

        for ei in range(num_experts):
            ax.plot(epochs, load_matrix[:, ei], marker='s', markersize=2, linewidth=1.4,
                    label=expert_names[ei])
        if first_ax_for_legend is None:
            first_ax_for_legend = ax
        ax.set_title(f'α={alpha_tag}', fontsize=11)
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.set_ylabel('Expert load', fontsize=9)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    for c in range(ncols):
        ax = axes[nrows - 1, c]
        if ax.get_visible():
            ax.set_xlabel('Epoch', fontsize=9)

    if first_ax_for_legend is not None:
        handles, labels = first_ax_for_legend.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)),
                       fontsize=9, bbox_to_anchor=(0.5, 1.0), frameon=True)

    fig.suptitle('Expert selection vs epoch (mean load over MoE layers)', fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ========== Figure 2: Compare entropy curves across different alpha values ==========

def plot_compare(routing_dirs, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stable order: ascending noise scale α (same metrics as before, clearer legend)
    def _sort_key(d):
        a = routing_alpha_from_path(d)
        return (a is not None, a if a is not None else 0.0, d)

    routing_dirs = sorted(routing_dirs, key=_sort_key)

    for routing_dir in routing_dirs:
        alpha_str = routing_dir.rstrip('/').split('_')[-1]
        snapshots = load_all_snapshots(routing_dir)
        if not snapshots:
            print(f"Warning: no routing_epoch_*.pt in {routing_dir}, skipped")
            continue

        epochs = [s['epoch'] for s in snapshots]
        num_modules = len(snapshots[0]['modules'])

        avg_entropies = []
        for s in snapshots:
            avg_e = np.mean([s['modules'][mi]['entropy'] for mi in range(num_modules)])
            avg_entropies.append(avg_e)
        axes[0].plot(epochs, avg_entropies, marker='o', markersize=4, label=f'α={alpha_str}')

        num_experts = len(snapshots[0]['modules'][0]['load'])
        load_stds = []
        for s in snapshots:
            loads_per_expert = []
            for ei in range(num_experts):
                avg_load = np.mean([s['modules'][mi]['load'][ei].item() for mi in range(num_modules)])
                loads_per_expert.append(avg_load)
            load_stds.append(np.std(loads_per_expert))
        axes[1].plot(epochs, load_stds, marker='s', markersize=4, label=f'α={alpha_str}')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Router Entropy')
    axes[0].set_title('Router Entropy vs. α')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Expert Load Std (lower = more balanced)')
    axes[1].set_title('Expert Load Balance vs. α')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ========== Figure 3: Per-pixel expert assignment heatmap ==========

def plot_heatmap(routing_dir, epoch, save_path=None):
    snapshot_path = os.path.join(routing_dir, f'routing_epoch_{epoch}.pt')
    if not os.path.exists(snapshot_path):
        print(f"Snapshot not found: {snapshot_path}")
        return

    snapshot = load_snapshot(snapshot_path)
    if 'input' not in snapshot:
        print(f"Snapshot is lite mode (no input/gating data). Use --mode dynamics or compare instead.")
        return
    input_imgs = snapshot['input']       # [B, 3, H, W]
    labels = snapshot['labels']          # [B, 1, H, W]
    modules = snapshot['modules']

    img_idx = 0
    img = input_imgs[img_idx].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    label = labels[img_idx, 0].numpy()

    num_modules = len(modules)
    cols_per_module = 2
    fig = plt.figure(figsize=(5 * (num_modules * cols_per_module + 2), 5))
    gs = GridSpec(1, num_modules * cols_per_module + 2, figure=fig)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_title('Input Image')
    ax_img.axis('off')

    ax_lbl = fig.add_subplot(gs[0, 1])
    ax_lbl.imshow(label, cmap='hot')
    ax_lbl.set_title('Ground Truth')
    ax_lbl.axis('off')

    num_experts = modules[0]['gating'].shape[1]
    cmap = plt.cm.get_cmap('tab10', num_experts)

    for mi, mod in enumerate(modules):
        indices = mod['indices'][img_idx]   # [topk, H', W'] (H' = H/patch_size)
        gating = mod['gating'][img_idx]     # [num_experts, H', W']
        patch_size = int(mod.get('patch_size', 1))

        dominant_expert = indices[0].numpy()
        h_mod, w_mod = dominant_expert.shape
        if patch_size > 1:
            # Up-sample patch-level decisions back to input resolution so the
            # heat-map aligns with the input image / GT mask.
            dominant_expert = np.kron(dominant_expert, np.ones((patch_size, patch_size), dtype=dominant_expert.dtype))
        short_name = mod['name'].split('.')[-2] if '.' in mod['name'] else mod['name']

        ax_assign = fig.add_subplot(gs[0, 2 + mi * cols_per_module])
        im = ax_assign.imshow(dominant_expert, cmap=cmap, vmin=0, vmax=num_experts - 1, interpolation='nearest')
        title = f'{short_name}\nExpert Map ({h_mod}x{w_mod})'
        if patch_size > 1:
            title += f', p={patch_size}'
        ax_assign.set_title(title, fontsize=9)
        ax_assign.axis('off')

        ax_entropy = fig.add_subplot(gs[0, 3 + mi * cols_per_module])
        entropy_map = -(gating * (gating + 1e-8).log()).sum(dim=0).numpy()
        if patch_size > 1:
            entropy_map = np.kron(entropy_map, np.ones((patch_size, patch_size), dtype=entropy_map.dtype))
        ax_entropy.imshow(entropy_map, cmap='viridis', interpolation='nearest')
        ax_entropy.set_title(f'{short_name}\nEntropy Map', fontsize=9)
        ax_entropy.axis('off')

    plt.suptitle(f'Routing Visualization (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ========== Figure 4: Evolution of expert maps across epochs ==========

def plot_evolution(routing_dir, module_idx=0, save_path=None):
    snapshots = load_all_snapshots(routing_dir)
    if not snapshots:
        print(f"No snapshots found in {routing_dir}")
        return

    if 'gating' not in snapshots[0]['modules'][module_idx]:
        print(f"Snapshots are lite mode (no gating data). Use --mode dynamics or compare instead.")
        return

    num_experts = snapshots[0]['modules'][module_idx]['gating'].shape[1]
    cmap = plt.cm.get_cmap('tab10', num_experts)

    n = len(snapshots)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for si, snap in enumerate(snapshots):
        mod = snap['modules'][module_idx]
        epoch = snap['epoch']
        indices = mod['indices'][0]     # [topk, H, W]
        gating = mod['gating'][0]       # [num_experts, H, W]

        dominant = indices[0].numpy()
        axes[0, si].imshow(dominant, cmap=cmap, vmin=0, vmax=num_experts - 1, interpolation='nearest')
        axes[0, si].set_title(f'Epoch {epoch}', fontsize=10)
        axes[0, si].axis('off')

        entropy = -(gating * (gating + 1e-8).log()).sum(dim=0).numpy()
        im = axes[1, si].imshow(entropy, cmap='viridis', interpolation='nearest')
        axes[1, si].set_title(f'Entropy (mean={entropy.mean():.3f})', fontsize=9)
        axes[1, si].axis('off')

    axes[0, 0].set_ylabel('Expert Assignment', fontsize=11)
    axes[1, 0].set_ylabel('Routing Entropy', fontsize=11)

    short_name = snapshots[0]['modules'][module_idx]['name']
    plt.suptitle(f'Routing Evolution — {short_name}', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MoE routing statistics')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['dynamics', 'compare', 'expert_loads', 'expert_loads_grid', 'heatmap', 'evolution', 'export', 'routing_png'],
                        help='dynamics: entropy/load curves; compare: multi-alpha comparison; '
                             'expert_loads: four experts’ load vs epoch (single routing_dir); '
                             'expert_loads_grid: same for every α (subplots, --run_roots or --routing_dirs); '
                             'heatmap: per-pixel expert map; evolution: expert map across epochs; '
                             'export: dump test samples + PT + routing_vis PNGs; '
                             'routing_png: convert existing routing.pt to PNG (--routing_export_dir or --routing_pt)')
    parser.add_argument('--routing_dir', type=str, default=None,
                        help='Path to routing log directory (for dynamics/heatmap/evolution)')
    parser.add_argument('--routing_dirs', type=str, nargs='+', default=None,
                        help='Paths to multiple routing log directories (for compare)')
    parser.add_argument('--run_roots', type=str, nargs='+', default=None,
                        help='Experiment directories each containing one routing_alpha_* folder (for compare)')
    parser.add_argument('--epoch', type=int, default=999,
                        help='Epoch to visualize (for heatmap mode)')
    parser.add_argument('--module_idx', type=int, default=0,
                        help='Which MoE module to visualize (for evolution mode)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save path for the figure (if not set, shows interactively)')
    parser.add_argument('--expert_grid_ncols', type=int, default=4,
                        help='Number of columns in expert_loads_grid subplot layout')

    # --- export mode (checkpoint + test loader) ---
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pth.tar / .pth (export mode)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for export mode')
    parser.add_argument('--start_batch', type=int, default=0,
                        help='Skip this many test DataLoader batches before saving (export)')
    parser.add_argument('--num_images', type=int, default=10,
                        help='How many test samples to save this run (export)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Test loader batch size (export); use 1 so each batch is one image')
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='Dataset name for TestSetLoader (export)')
    parser.add_argument('--root', type=str, default='dataset/',
                        help='Dataset root containing <dataset>/ (export)')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='Split folder name (export)')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='Image suffix (export)')
    parser.add_argument('--data_mode', type=str, default='TXT',
                        choices=['TXT', 'SIATD10seq'],
                        help='TXT: load_dataset; SIATD10seq: load_dataset_5folders (export)')
    parser.add_argument('--channel_size', type=str, default='three',
                        help='DNANet channel_size (export)')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='Backbone tag for num_blocks (export)')
    parser.add_argument('--moe_stages', type=str, default='1,1,1,1',
                        help='Comma 0/1 flags per stage (export)')
    parser.add_argument('--dilations', type=str, default='1,2,2,3',
                        help='Comma-separated dilations (export)')
    parser.add_argument('--noise_scale', type=float, default=0.2,
                        help='Router noise scale α (export, must match training)')
    parser.add_argument('--patch_size', type=str, default='1',
                        help='MoE patch size: one int or comma list (export)')
    parser.add_argument('--top_k', type=int, default=2,
                        help='MoE top-k (export)')
    parser.add_argument('--input_routing', type=str, default='full',
                        choices=['full', 'sparse'],
                        help='MoE input routing (export)')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Model input channels (export)')
    parser.add_argument('--device', type=str, default='auto',
                        help='cuda | cpu | auto (export)')
    parser.add_argument('--workers', type=int, default=0,
                        help='DataLoader workers (export and optional)')
    parser.add_argument('--base_size', type=int, default=256,
                        help='export: TestSetLoader resize edge (same as train/test eval; no random augment)')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='export: kept for API parity with Trainer; TestSetLoader eval uses base_size only')
    parser.add_argument('--routing_export_dir', type=str, default=None,
                        help='routing_png: recursively find routing.pt under this directory')
    parser.add_argument('--routing_pt', type=str, nargs='*', default=None,
                        help='routing_png: one or more routing.pt file paths')

    args = parser.parse_args()

    if args.mode == 'dynamics':
        plot_dynamics(args.routing_dir, save_path=args.save)
    elif args.mode == 'expert_loads':
        if not args.routing_dir:
            parser.error('expert_loads mode requires --routing_dir')
        plot_expert_loads(args.routing_dir, save_path=args.save)
    elif args.mode == 'expert_loads_grid':
        if args.run_roots and args.routing_dirs:
            parser.error('Use either --routing_dirs or --run_roots for expert_loads_grid, not both')
        if args.run_roots:
            routing_dirs = discover_routing_dirs_from_run_roots(args.run_roots)
            if not routing_dirs:
                parser.error('No routing directories discovered from --run_roots')
            plot_expert_loads_grid(routing_dirs, save_path=args.save, ncols=args.expert_grid_ncols)
        elif args.routing_dirs:
            plot_expert_loads_grid(args.routing_dirs, save_path=args.save, ncols=args.expert_grid_ncols)
        else:
            parser.error('expert_loads_grid mode requires --routing_dirs or --run_roots')
    elif args.mode == 'compare':
        if args.run_roots and args.routing_dirs:
            parser.error('Use either --routing_dirs or --run_roots for compare, not both')
        if args.run_roots:
            routing_dirs = discover_routing_dirs_from_run_roots(args.run_roots)
            if not routing_dirs:
                parser.error('No routing directories discovered from --run_roots')
            plot_compare(routing_dirs, save_path=args.save)
        elif args.routing_dirs:
            plot_compare(args.routing_dirs, save_path=args.save)
        else:
            parser.error('compare mode requires --routing_dirs or --run_roots')
    elif args.mode == 'heatmap':
        plot_heatmap(args.routing_dir, args.epoch, save_path=args.save)
    elif args.mode == 'evolution':
        plot_evolution(args.routing_dir, module_idx=args.module_idx, save_path=args.save)
    elif args.mode == 'export':
        if not args.checkpoint or not args.out_dir:
            parser.error('export mode requires --checkpoint and --out_dir')
        export_test_routing(
            args.checkpoint,
            args.out_dir,
            start_batch=args.start_batch,
            num_images=args.num_images,
            test_batch_size=args.test_batch_size,
            dataset=args.dataset,
            root=args.root,
            split_method=args.split_method,
            suffix=args.suffix,
            data_mode=args.data_mode,
            workers=args.workers,
            base_size=args.base_size,
            crop_size=args.crop_size,
            channel_size=args.channel_size,
            backbone=args.backbone,
            moe_stages_str=args.moe_stages,
            dilations_str=args.dilations,
            noise_scale=args.noise_scale,
            patch_size_str=args.patch_size,
            top_k=args.top_k,
            input_routing=args.input_routing,
            in_channels=args.in_channels,
            device=args.device,
        )
    elif args.mode == 'routing_png':
        if not args.routing_export_dir and not args.routing_pt:
            parser.error('routing_png requires --routing_export_dir and/or --routing_pt')
        if args.routing_pt:
            for p in args.routing_pt:
                if not os.path.isfile(p):
                    parser.error(f'Not a file: {p}')
                routing_pt_to_pngs(p)
        if args.routing_export_dir:
            if not os.path.isdir(args.routing_export_dir):
                parser.error(f'Not a directory: {args.routing_export_dir}')
            routing_export_dir_to_pngs(args.routing_export_dir)
