#!/usr/bin/env python
"""Build the selected m11 router visualization figure for IRSTD."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


SAMPLES = [
    "ds00000_XDU189",
    "ds00002_XDU672",
    "ds00005_XDU888",
    "ds00013_XDU637",
    "ds00019_XDU117",
    "ds00026_XDU54",
    "ds00034_XDU193",
]

COLUMNS = [
    ("image", "input_rgb.png"),
    ("label", "label.png"),
    ("entropy", "routing_vis/m11_attn_entropy.png"),
    ("expert map", "routing_vis/m11_attn_expert_map.png"),
    ("context", "routing_vis/m11_attn_gating_e0.png"),
    ("spatial", "routing_vis/m11_attn_gating_e1.png"),
    ("Identity", "routing_vis/m11_attn_gating_e2.png"),
]


def read_image(path):
    img = Image.open(path)
    if img.mode == "L":
        return np.asarray(img), "gray"
    return np.asarray(img.convert("RGB")), None


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


def crop_window(box, height, width):
    x0, y0, x1, y1 = box
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def _rect_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))


def _box_to_axes_rect(box, height, width):
    x0, y0, x1, y1 = box
    return x0 / width, 1.0 - y1 / height, x1 / width, 1.0 - y0 / height


def _choose_inset_rect(box, all_boxes, used_rects, height, width, size):
    margin = 0.025
    starts = [
        (1 - margin - size, 1 - margin - size),
        (margin, 1 - margin - size),
        (1 - margin - size, margin),
        (margin, margin),
        (1 - margin - size, 0.5 - size / 2),
        (margin, 0.5 - size / 2),
    ]
    target_rects = [_box_to_axes_rect(b, height, width) for b in all_boxes]
    bx0, by0, bx1, by1 = _box_to_axes_rect(box, height, width)
    bc_x, bc_y = (bx0 + bx1) / 2, (by0 + by1) / 2

    best = None
    best_score = None
    for left, bottom in starts:
        cand = (left, bottom, left + size, bottom + size)
        target_overlap = sum(_rect_overlap(cand, r) for r in target_rects)
        used_overlap = sum(_rect_overlap(cand, r) for r in used_rects)
        cc_x, cc_y = left + size / 2, bottom + size / 2
        dist = ((cc_x - bc_x) ** 2 + (cc_y - bc_y) ** 2) ** 0.5
        score = target_overlap * 1000 + used_overlap * 200 - dist
        if best_score is None or score < best_score:
            best_score = score
            best = cand
    return best


def draw_boxes_and_insets(ax, img, cmap, boxes):
    h, w = img.shape[:2]
    inset_size = 0.34 if len(boxes) == 1 else 0.27
    used_rects = []

    for box in boxes:
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="#ff2d2d",
            linewidth=1.15,
        ))

    for box in boxes:
        x0, y0, x1, y1 = box
        cx0, cy0, cx1, cy1 = crop_window(box, h, w)
        crop = img[cy0:cy1, cx0:cx1]
        inset_rect = _choose_inset_rect(box, boxes, used_rects, h, w, inset_size)
        used_rects.append(inset_rect)

        left, bottom, right, top = inset_rect
        iax = ax.inset_axes([left, bottom, right - left, top - bottom])
        iax.imshow(crop, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        iax.set_xticks([])
        iax.set_yticks([])
        iax.add_patch(Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="#ff2d2d",
            linewidth=1.25,
            transform=iax.transAxes,
            clip_on=False,
        ))
        for spine in iax.spines.values():
            spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", default="routing_export/IRSTD")
    parser.add_argument("--out", default="routing_export/IRSTD/router_m11_effect_figure.png")
    parser.add_argument("--pdf", default="routing_export/IRSTD/router_m11_effect_figure.pdf")
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    nrows, ncols = len(SAMPLES), len(COLUMNS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 10.7), dpi=args.dpi)

    for col, (title, _) in enumerate(COLUMNS):
        axes[0, col].set_title(title, fontsize=15, pad=10)

    for row, sample in enumerate(SAMPLES):
        sample_dir = os.path.join(args.export_dir, sample)
        boxes = target_boxes_from_label(os.path.join(sample_dir, "label.png"))
        for col, (_, rel_path) in enumerate(COLUMNS):
            ax = axes[row, col]
            img, cmap = read_image(os.path.join(sample_dir, rel_path))
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
            draw_boxes_and_insets(ax, img, cmap, boxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01, wspace=0.015, hspace=0.015)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    fig.savefig(args.pdf)
    plt.close(fig)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.pdf}")


if __name__ == "__main__":
    main()
