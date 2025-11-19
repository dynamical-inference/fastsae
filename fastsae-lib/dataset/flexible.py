from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import inspect
import os
from typing import Any, Callable, Optional

from datasets import Dataset as HFDatasetType
from datasets import load_dataset
from datasets import load_from_disk
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


def _hf_transform(
    batch: dict[str, Any],
    *,
    processor: Optional[Callable] = None,
    image_key: str = "image",
    label_key: str = "label",
    return_label: bool = False,
    return_pixel_values_only: bool = False,
) -> dict[str, Any]:
    image_or_item = batch[image_key]
    is_batched = isinstance(image_or_item, (list, tuple))

    # Optionally extract label(s)
    label_value: Optional[Any] = None
    if return_label and label_key in batch:
        label_value = batch[label_key]

    # --- no processor ---
    if processor is None:
        if return_label:
            return {"pixel_values": image_or_item, "label": label_value}
        return {"pixel_values": image_or_item}

    # --- apply processor ---
    try:
        sig = inspect.signature(processor)
        if "text" in sig.parameters and "images" in sig.parameters:
            out = processor(
                images=image_or_item,
                text=["a photo"] * len(image_or_item) if is_batched else "a photo",
                return_tensors="pt",
            )
        elif "images" in sig.parameters:
            out = processor(images=image_or_item, return_tensors="pt")
        else:
            out = processor(image_or_item)
    except Exception:
        out = processor(image_or_item)

    if isinstance(out, Mapping):
        out = dict(out)

    # --- batched mode from HF set_transform ---
    if is_batched and isinstance(out, dict):
        first_key = next(
            (k for k, v in out.items() if isinstance(v, torch.Tensor) and v.ndim > 0),
            None,
        )
        batch_dim = out[first_key].shape[0] if first_key is not None else len(image_or_item)

        # ✅ handle return_pixel_values_only=True separately (simpler + faster)
        if return_pixel_values_only and "pixel_values" in out:
            pv = out["pixel_values"]
            if return_label:
                if isinstance(label_value, (list, tuple)) and len(label_value) == batch_dim:
                    return {"pixel_values": pv, "label": torch.as_tensor(label_value)}
                else:
                    return {
                        "pixel_values": pv,
                        "label": torch.full((batch_dim,), label_value),
                    }
            return {"pixel_values": pv}

        # otherwise return full dict (default behavior)
        combined = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == batch_dim:
                combined[k] = [v[i] for i in range(batch_dim)]
            else:
                combined[k] = [v] * batch_dim

        if return_label:
            if isinstance(label_value, (list, tuple)) and len(label_value) == batch_dim:
                combined["label"] = label_value
            else:
                combined["label"] = [label_value] * batch_dim

        return combined

    # --- single item mode ---
    if isinstance(out, dict):
        if "pixel_values" in out and return_pixel_values_only:
            pv = out["pixel_values"]
            if return_label:
                return {"pixel_values": pv, "label": label_value}
            return {"pixel_values": pv}
        if return_label and ("label" not in out and "labels" not in out):
            out["label"] = label_value
        return out

    # --- fallback ---
    if return_label:
        return {"pixel_values": out, "label": label_value}
    return {"pixel_values": out}


def collate_auto(batch: list[Any]) -> Any:
    """
    Heuristic collate:
      - Tensor or dict of tensors → default_collate
      - Otherwise (e.g., PIL images) → return as a list
    """
    first = batch[0]
    if isinstance(first, torch.Tensor):
        return default_collate(batch)
    if isinstance(first, dict):
        # Let default_collate handle dict[str, Tensor]
        return default_collate(batch)
    # Fallback: return as-is (list of items, e.g., PIL.Image)
    return batch


def collate_images_only(batch: list[Any]) -> list[Any]:
    """Simple collate that returns the batch list unchanged (useful for PIL images)."""
    return batch


def create_dataloader(
    dataset: str | Any,
    *,
    processor: Optional[Callable] = None,
    return_pixel_values_only: bool = True,  # True when train, False when classifiation etc # TODO: fix batch_size bug when False
    return_label: bool = False,
    return_labels_np: bool = False,  # True when compute_stats
    label_key: str = "label",
    image_key: str = "image",
    split: str = "train",
    shuffle: bool = False,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    collate: str = "auto",  # "auto" | "default" | "images"
    hf_home_env: str = "HF_HOME",
    hf_name_map: Optional[dict[str, str]] = None,
) -> DataLoader:
    """
    Unified DataLoader factory for Hugging Face datasets.

    - Accepts an HF dataset name, a path to a saved HF dataset, or a `datasets.Dataset` object.
    - Applies a processor on-the-fly using `datasets.Dataset.set_transform`.
    - Auto-selects a safe collate function based on the processed item type.

    Args:
        dataset: HF dataset name (requires `hf_name_map`), path to a saved dataset, or a `datasets.Dataset` object.
        processor: Callable to apply to each dataset item.
        return_pixel_values_only: If processor returns a dict, only return the value for `pixel_values`.
        return_label: Whether to include the label in the output.
        return_labels_np: Whether to return the labels as a numpy array.
        label_key: The key for the label in the dataset item.
        image_key: The key for the image in the dataset item.
        split: The dataset split to use.
        batch_size: DataLoader batch size.
        num_workers: DataLoader num_workers.
        pin_memory: DataLoader pin_memory.
        persistent_workers: DataLoader persistent_workers.
        collate: Collate function type ("auto", "default", "images").
        hf_home_env: Environment variable for HF cache directory.
        hf_name_map: Mapping from short names to HF dataset names.
    """
    ds_obj: Any
    if isinstance(dataset, str):
        if os.path.isdir(dataset):
            ds_obj = load_from_disk(dataset)
            if isinstance(ds_obj, dict) and split in ds_obj:
                ds_obj = ds_obj[split]

        else:
            if hf_name_map is None:
                raise ValueError("hf_name_map must be provided when dataset is a string.")
            if dataset not in hf_name_map:
                raise KeyError(f"Unknown dataset key '{dataset}'. Provide it in hf_name_map.")
            ds_name = hf_name_map[dataset]
            ds_obj = load_dataset(ds_name, cache_dir=os.getenv(hf_home_env), split=split)
    elif isinstance(dataset, HFDatasetType):
        ds_obj = dataset

    else:
        raise NotImplementedError(f"Dataset must be a string or a datasets.Dataset object. Got {type(dataset)}.")

    labels_np = None
    if return_labels_np:
        labels_np = np.array(ds_obj[label_key])

    transform = partial(
        _hf_transform,
        processor=processor,
        image_key=image_key,
        label_key=label_key,
        return_label=return_label,
        return_pixel_values_only=return_pixel_values_only,
    )
    wrapped = ds_obj.with_transform(transform)

    # Decide collate function
    if collate == "default":
        collate_fn = default_collate
    elif collate == "images":
        collate_fn = collate_images_only
    elif collate == "auto":
        # Peek one sample to choose a safe collate function
        try:
            sample = wrapped[0]
            if isinstance(sample, (torch.Tensor, dict)):
                collate_fn = default_collate
            else:
                collate_fn = collate_auto
        except Exception:
            collate_fn = collate_auto
    else:
        raise ValueError("collate must be one of {'auto','default','images'}")

    transformed_dataloader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn,
    )

    if return_labels_np:
        return transformed_dataloader, labels_np
    else:
        return transformed_dataloader


__all__ = [
    "collate_auto",
    "collate_images_only",
    "create_dataloader",
]
