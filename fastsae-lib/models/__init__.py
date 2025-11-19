from .backbone import Backbone
from .sae import VanillaSAE
from .wrapper import SAEWrapper

__all__ = ["Backbone", "SAEWrapper"]


def SAE(**kwargs):
    """
    Factory that returns an SAE variant instance based on the `architecture` argument.
    Usage remains: from sae.models import SAE; model = SAE(..., architecture="base")
    """
    arch = (kwargs.get("architecture") or "vanilla").lower()
    if arch == "vanilla":
        return VanillaSAE(**kwargs)

    raise ValueError(f"Unknown SAE architecture: {arch}")


def _sae_load(load_dir):
    """
    Load an SAE instance from a checkpoint directory by inspecting the saved
    config to determine the correct SAE subclass, then delegating to that
    class's load().
    """
    import json
    import pathlib

    load_path = pathlib.Path(load_dir)
    if not load_path.exists():
        raise FileNotFoundError(f"Load path not found: {load_path}")

    # Heuristic: find the SAE config JSON among any JSON files in the folder.
    # Prefer files that contain keys typical for SAE configs.
    cfg = None
    for json_file in sorted(load_path.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and ("d_in" in data) and ("class_type" in data or "architecture" in data):
            cfg = data
            break

    if cfg is None:
        raise FileNotFoundError(f"Could not locate SAE config JSON in {load_path}. Expected a *.json with 'd_in' and 'class_type' or 'architecture'.")

    class_type = cfg.get("class_type")
    architecture = (cfg.get("architecture") or "").lower()

    # Map both explicit class_type and architecture to their classes
    class_by_name = {
        "VanillaSAE": VanillaSAE,
    }

    class_by_arch = {
        "vanilla": VanillaSAE,
    }

    target_cls = None
    if isinstance(class_type, str) and class_type in class_by_name:
        target_cls = class_by_name[class_type]
    elif architecture in class_by_arch:
        target_cls = class_by_arch[architecture]

    if target_cls is None:
        raise ValueError(f"Unknown SAE class from config. class_type={class_type!r}, architecture={architecture!r}")

    return target_cls.load(load_path)


# Attach as a staticmethod-like attribute so callers can do: SAE.load(dir)
SAE.load = staticmethod(_sae_load)


__all__ = [
    "SAE",
]
