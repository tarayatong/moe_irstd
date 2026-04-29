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
                        choices=['dynamics', 'compare', 'expert_loads', 'expert_loads_grid', 'heatmap', 'evolution'],
                        help='dynamics: entropy/load curves; compare: multi-alpha comparison; '
                             'expert_loads: four experts’ load vs epoch (single routing_dir); '
                             'expert_loads_grid: same for every α (subplots, --run_roots or --routing_dirs); '
                             'heatmap: per-pixel expert map; evolution: expert map across epochs')
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
