#!/usr/bin/env python
"""Build a 4x4 expert-map overview for one exported routing sample."""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


MODULES = [
    (0, 1, "m00", (256, 256)),
    (0, 2, "m01", (128, 128)),
    (0, 3, "m02", (64, 64)),
    (1, 0, "m03", (512, 512)),
    (1, 1, "m04", (256, 256)),
    (1, 2, "m05", (128, 128)),
    (1, 3, "m06", (64, 64)),
    (2, 0, "m07", (512, 512)),
    (2, 1, "m08", (256, 256)),
    (2, 2, "m09", (128, 128)),
    (2, 3, "m10", (64, 64)),
    (3, 0, "m11", (512, 512)),
    (3, 1, "m12", (256, 256)),
    (3, 2, "m13", (128, 128)),
    (3, 3, "m14", (64, 64)),
]

EXPERT_NAMES = ["E0: context", "E1: spatial", "E2: Identity"]


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _component_boxes(binary):
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    boxes = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs, ys = [], []
            while stack:
                sx, sy = stack.pop()
                xs.append(sx)
                ys.append(sy)
                for ny in range(max(0, sy - 1), min(h, sy + 2)):
                    for nx in range(max(0, sx - 1), min(w, sx + 2)):
                        if binary[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            boxes.append((min(xs), min(ys), max(xs), max(ys), len(xs)))
    return boxes


def target_boxes_from_label(label_path, margin=6, min_size=18):
    label = np.asarray(Image.open(label_path).convert("L"))
    binary = label > 0
    components = _component_boxes(binary)
    if not components:
        h, w = label.shape
        return [(0, 0, w - 1, h - 1)]

    boxes = []
    h, w = label.shape
    for x0, y0, x1, y1, area in sorted(components, key=lambda b: (b[1], b[0])):
        if area <= 0:
            continue
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        half_w = max((x1 - x0 + 1) // 2, min_size // 2)
        half_h = max((y1 - y0 + 1) // 2, min_size // 2)
        boxes.append((
            max(0, cx - half_w - margin),
            max(0, cy - half_h - margin),
            min(w, cx + half_w + margin + 1),
            min(h, cy + half_h + margin + 1),
        ))
    return boxes


def scale_boxes(boxes, src_hw, dst_hw):
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = dst_w / src_w
    sy = dst_h / src_h
    return [
        (
            int(round(x0 * sx)),
            int(round(y0 * sy)),
            int(round(x1 * sx)),
            int(round(y1 * sy)),
        )
        for x0, y0, x1, y1 in boxes
    ]


def _choose_far_corner(box, height, width, size):
    margin = 0.025
    corners = [
        (margin, 1 - margin - size),
        (1 - margin - size, 1 - margin - size),
        (margin, margin),
        (1 - margin - size, margin),
    ]
    x0, y0, x1, y1 = box
    tx = ((x0 + x1) / 2) / width
    ty = 1.0 - ((y0 + y1) / 2) / height
    return max(
        corners,
        key=lambda p: ((p[0] + size / 2 - tx) ** 2 + (p[1] + size / 2 - ty) ** 2),
    )


def draw_boxes_and_insets(ax, img, boxes):
    h, w = img.shape[:2]
    inset_size = 0.34 if len(boxes) == 1 else 0.27

    for box in boxes:
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="white",
            linewidth=1.2,
        ))

    for box in boxes:
        x0, y0, x1, y1 = box
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, x1), min(h, y1)
        crop = img[cy0:cy1, cx0:cx1]
        left, bottom = _choose_far_corner(box, h, w, inset_size)
        iax = ax.inset_axes([left, bottom, inset_size, inset_size])
        iax.imshow(crop, interpolation="nearest")
        iax.set_xticks([])
        iax.set_yticks([])
        iax.add_patch(Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="white",
            linewidth=1.4,
            transform=iax.transAxes,
            clip_on=False,
        ))
        for spine in iax.spines.values():
            spine.set_visible(False)


def draw_panel(ax, img, label, boxes):
    ax.imshow(img, interpolation="nearest")
    draw_boxes_and_insets(ax, img, boxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.025,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=8.5,
        linespacing=1.15,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.62,
        },
    )


def add_expert_legend(fig):
    cmap = plt.cm.get_cmap("tab10", max(len(EXPERT_NAMES), 3))
    norm = mcolors.Normalize(vmin=0, vmax=max(len(EXPERT_NAMES) - 1, 1))
    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=cmap(norm(idx)),
            edgecolor="black",
            linewidth=0.8,
        )
        for idx in range(len(EXPERT_NAMES))
    ]
    legend = fig.legend(
        handles,
        EXPERT_NAMES,
        loc="lower center",
        ncol=len(EXPERT_NAMES),
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        columnspacing=1.6,
        borderpad=0.55,
    )
    for text in legend.get_texts():
        text.set_color("black")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_dir",
        default="routing_export/IRSTD/ds00000_XDU189",
        help="Exported sample directory containing input_rgb.png and routing_vis/.",
    )
    parser.add_argument(
        "--out",
        default="routing_export/IRSTD/ds00000_XDU189/expert_map_4x4.png",
    )
    parser.add_argument(
        "--pdf",
        default="routing_export/IRSTD/ds00000_XDU189/expert_map_4x4.pdf",
    )
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    fig, axes = plt.subplots(4, 4, figsize=(8.0, 8.0), dpi=args.dpi)
    axes = axes.reshape(-1)

    input_path = os.path.join(args.sample_dir, "input_rgb.png")
    label_path = os.path.join(args.sample_dir, "label.png")
    image_boxes = target_boxes_from_label(label_path)
    draw_panel(axes[0], read_rgb(input_path), "stage 0, block 0\n512 x 512", image_boxes)

    for ax, (stage, block, module, (height, width)) in zip(axes[1:], MODULES):
        map_path = os.path.join(args.sample_dir, "routing_vis", f"{module}_attn_expert_map.png")
        label = f"stage {stage}, block {block}\n{height} x {width}"
        img = read_rgb(map_path)
        boxes = scale_boxes(image_boxes, (512, 512), img.shape[:2])
        draw_panel(ax, img, label, boxes)

    add_expert_legend(fig)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.055, wspace=0.01, hspace=0.01)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    fig.savefig(args.pdf)
    plt.close(fig)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.pdf}")


if __name__ == "__main__":
    main()
