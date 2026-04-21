"""
Routing visualization for SpatialSparseMoE ablation study on noise scale alpha.

Usage:
    # 1. Plot routing dynamics across training epochs for a single alpha
    python visualize_routing.py --mode dynamics --routing_dir result/routing_alpha_0.2

    # 2. Compare routing dynamics across multiple alpha values
    python visualize_routing.py --mode compare --routing_dirs result/routing_alpha_0.0 result/routing_alpha_0.1 result/routing_alpha_0.2 result/routing_alpha_0.5

    # 3. Visualize per-pixel expert assignment on test images
    python visualize_routing.py --mode heatmap --routing_dir result/routing_alpha_0.2 --epoch 999
"""

import argparse
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


def load_snapshot(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_all_snapshots(routing_dir):
    files = sorted(glob.glob(os.path.join(routing_dir, 'routing_epoch_*.pt')),
                   key=lambda f: int(f.split('_')[-1].replace('.pt', '')))
    snapshots = []
    for f in files:
        snapshots.append(load_snapshot(f))
    return snapshots


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
    expert_names = [f'Expert {i}' for i in range(num_experts)]
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


# ========== Figure 2: Compare entropy curves across different alpha values ==========

def plot_compare(routing_dirs, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for routing_dir in routing_dirs:
        alpha_str = routing_dir.rstrip('/').split('_')[-1]
        snapshots = load_all_snapshots(routing_dir)
        if not snapshots:
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
        indices = mod['indices'][img_idx]   # [topk, H, W]
        gating = mod['gating'][img_idx]     # [num_experts, H, W]

        dominant_expert = indices[0].numpy()
        h_mod, w_mod = dominant_expert.shape
        short_name = mod['name'].split('.')[-2] if '.' in mod['name'] else mod['name']

        ax_assign = fig.add_subplot(gs[0, 2 + mi * cols_per_module])
        im = ax_assign.imshow(dominant_expert, cmap=cmap, vmin=0, vmax=num_experts - 1, interpolation='nearest')
        ax_assign.set_title(f'{short_name}\nExpert Map ({h_mod}x{w_mod})', fontsize=9)
        ax_assign.axis('off')

        ax_entropy = fig.add_subplot(gs[0, 3 + mi * cols_per_module])
        entropy_map = -(gating * (gating + 1e-8).log()).sum(dim=0).numpy()
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
                        choices=['dynamics', 'compare', 'heatmap', 'evolution'],
                        help='dynamics: entropy/load curves; compare: multi-alpha comparison; '
                             'heatmap: per-pixel expert map; evolution: expert map across epochs')
    parser.add_argument('--routing_dir', type=str, default=None,
                        help='Path to routing log directory (for dynamics/heatmap/evolution)')
    parser.add_argument('--routing_dirs', type=str, nargs='+', default=None,
                        help='Paths to multiple routing log directories (for compare)')
    parser.add_argument('--epoch', type=int, default=999,
                        help='Epoch to visualize (for heatmap mode)')
    parser.add_argument('--module_idx', type=int, default=0,
                        help='Which MoE module to visualize (for evolution mode)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save path for the figure (if not set, shows interactively)')
    args = parser.parse_args()

    if args.mode == 'dynamics':
        plot_dynamics(args.routing_dir, save_path=args.save)
    elif args.mode == 'compare':
        plot_compare(args.routing_dirs, save_path=args.save)
    elif args.mode == 'heatmap':
        plot_heatmap(args.routing_dir, args.epoch, save_path=args.save)
    elif args.mode == 'evolution':
        plot_evolution(args.routing_dir, module_idx=args.module_idx, save_path=args.save)
