"""Convert an existing 3-expert context checkpoint; never overwrite inputs."""

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error("Output must be a new file")
    import torch
    from sfaoe.checkpoints import checkpoint_state_dict
    from sfaoe.config import parse_args
    from sfaoe.factory import build_model
    from model.load_param_data import load_param

    cfg = parse_args([], initialize_run=False)
    filters, blocks = load_param(cfg.channel_size, cfg.backbone)
    model = build_model(cfg, filters, blocks)
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    metadata = {
        "architecture": "context_spatial_identity_sparse",
        "expert_names": ["context", "spatial", "identity"],
        "input_routing": "sparse",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "conversion": "removed unused context llconv/hconv tensors only; no optimizer state",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save({"state_dict": model.state_dict(), "meta": metadata}, handle)
    print(args.output)


if __name__ == "__main__":
    main()
