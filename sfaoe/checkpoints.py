"""Strict loading of context/spatial/Identity checkpoints.

Old three-expert context runs registered unused llconv/hconv tensors inside
the context module. Only those dead tensors are removed; active tensors must
match strictly. Four-expert checkpoints cannot be converted by dropping HF.
"""

from collections import OrderedDict
import re


def clean_context_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Expected a state_dict mapping")
    clean = OrderedDict()
    for name, value in state.items():
        if ".spacial_moe.router." in name and name.endswith(("weight", "bias")):
            if value.shape[0] != 3:
                raise ValueError("Expected an existing 3-expert context checkpoint; four-expert weights need retraining")
        if re.search(r"\.spacial_moe\.experts\.0\.(?:llconv|hconv)\.", name):
            continue
        clean[name] = value
    return clean


def checkpoint_state_dict(checkpoint):
    """Accept plain state_dicts and the two historical checkpoint wrappers."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint:
                return clean_context_state_dict(checkpoint[key])
    return clean_context_state_dict(checkpoint)
