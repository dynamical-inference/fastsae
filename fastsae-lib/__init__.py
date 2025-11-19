"""
SAE (Sparse Autoencoder) Package

A package for efficient training of Sparse Autoencoders.
"""

__version__ = "0.1.2"

# Provide lazy access to common subpackages to avoid circular imports at import time.
# Usage preserved:
#   import fastsae; fastsae.models / fastsae.dataset / fastsae.training / fastsae.interpret / fastsae.sklearn
import importlib
from typing import Any

__all__ = [
    "models",
    "dataset",
    "training",
    "interpret",
    "sklearn",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name in {"models", "dataset", "training", "interpret", "sklearn"}:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
