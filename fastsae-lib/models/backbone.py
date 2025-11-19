from __future__ import annotations

import contextlib
import dataclasses
import inspect
import pathlib
from typing import Any, Dict, Optional

import config_dataclass
from fastsae.utils import verbose
import torch


class _EarlyStopForward(Exception):
    """Internal control-flow exception to abort forward after capturing the hook output."""

    pass


@config_dataclass.torch_dataclass
class Backbone(config_dataclass.ConfigurableTorchModule, torch.nn.Module):
    model_hf_name: str = config_dataclass.config_field(default=None)
    hook_target_name: str = config_dataclass.config_field(default=None)
    device: str = config_dataclass.config_field(default="cuda_if_available")
    _verbose: bool = dataclasses.field(default_factory=verbose.verbose_factory)

    def __lazy_post_init__(self):
        if self._verbose:
            print("🚀 Building Backbone (lazy initialization)...")

        try:
            import transformers

            self.model = transformers.AutoModel.from_pretrained(self.model_hf_name)  # Only supports huggingface models
        except Exception:
            raise ValueError("model_hf_name is not provided and model is not a huggingface model")

        self.dtype = torch.float32
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        self.model.eval()

        try:
            self.dtype = getattr(self.model, "dtype", None)
        except Exception:
            self.dtype = torch.float32

        self._cache: Optional[torch.Tensor] = None

        # Resolve exact module to hook
        self._hook_target = self._resolve_hook_target(self.model, self.hook_target_name)
        self._hook_target_dim = self.get_hook_target_dim()

        if self._verbose:
            print(f"Hook target dimension: {self._hook_target_dim}")

        self._image_size, self._patch_size = self.get_image_patch_size()

        if self._verbose:
            print("✅ Backbone built successfully!")

    # ---------- I/O: save / load ----------

    def _save_additional(self, path: pathlib.Path) -> None:
        """
        override to save model weights - always loaded from Huggingface.
        """
        print("Pass saving model weights - always loaded from Huggingface.")
        pass

    def _load_additional(self, path: pathlib.Path) -> None:
        """
        override to load model weights - always loaded from Huggingface.
        """
        print("Pass loading model weights - always loaded from Huggingface.")
        pass

    # ---------- Utilities ----------

    def _get_device(self):
        if self.device == "cuda_if_available":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def get_image_patch_size(self):
        # TODO: should I get image and patch size form processor?
        cfg = getattr(self.model, "config", None)
        vision_cfg = getattr(cfg, "vision_config", None) if cfg is not None else None

        # TODO: set default image and patch size
        image_size = None
        patch_size = None

        if vision_cfg is not None:
            image_size = getattr(vision_cfg, "image_size", None)
            patch_size = getattr(vision_cfg, "patch_size", None)

        # Fallback to top-level config if missing on vision_config
        if image_size is None and cfg is not None:
            image_size = getattr(cfg, "image_size", None)
        if patch_size is None and cfg is not None:
            patch_size = getattr(cfg, "patch_size", None)

        if image_size is None:
            image_size = 224
            print("Warning: Image size is not found in the model vision_config or config; using default image size 224.")
            # raise ValueError("Image size is not found in the model vision_config or config")
        if patch_size is None:
            patch_size = 16
            print("Warning: Patch size is not found in the model vision_config or config; using default patch size 16.")
            # raise ValueError("Patch size is not found in the model vision_config or config")

        # Ensure ints
        return int(image_size), int(patch_size)

    @torch.no_grad()
    def get_hook_target_dim(self):
        x = self._make_dummy_input()

        try:
            with self._hooked_model(replace_with=None, stop_after=True):  # capture up to hook
                _ = self._model_forward(x)
        except _EarlyStopForward:
            pass

        z = self._consume_cache()
        return z.shape[-1]

    @property
    def hook_target_dim(self):
        return self._hook_target_dim

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @torch.no_grad()
    def forward(self, x: Dict[str, torch.Tensor] | torch.Tensor, return_hooked_out: bool = False):
        """
        Run model; if return_hooked_out=True, capture the hook point’s output as z.
        Returns:
          (out, z) if return_hooked_out else out
        """
        with self._hooked_model(replace_with=None):  # capture mode
            out = self._model_forward(x)

        if return_hooked_out:
            z = self._consume_cache()
            return out, z
        return out

    @torch.no_grad()
    def forward_steered(self, x: Dict[str, torch.Tensor] | torch.Tensor, z_hat: torch.Tensor):
        """
        Re-run model while replacing hook point’s output with z_hat.
        """
        assert z_hat.shape[-1] == self._hook_target_dim, f"z_hat dimension mismatch: {z_hat.shape[-1]} != {self._hook_target_dim}"
        with self._hooked_model(replace_with=z_hat):
            out = self._model_forward(x)
        return out

    @torch.no_grad()
    def forward_to_z(self, x: Dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """
        Run the model only up to the hook point and stop early. Returns the hooked activation z.
        """
        if "latent" in x:
            return x["latent"].to(self.device)
        try:
            with self._hooked_model(replace_with=None, stop_after=True):
                z = self._model_forward(x)
        except _EarlyStopForward:
            pass
        out = self._consume_cache()
        if out is None:
            return z
        return out

    # ---------- Internals ----------

    def _model_forward(self, x: Dict[str, torch.Tensor] | torch.Tensor | list | tuple):
        """
        Hugging Face models usually accept dicts (e.g., {'pixel_values': ...} for vision models)
        or tensors depending on the class. Pass through as-is.
        """
        # Configure autocast for mixed precision when appropriate
        device_type = self.device
        use_autocast = isinstance(self.dtype, torch.dtype) and self.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        cm = torch.autocast(device_type=device_type, dtype=self.dtype) if use_autocast else contextlib.nullcontext()

        # Normalize list/tuple batches into a single dict/tensor batch
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("Received empty batch list/tuple")
            first = x[0]
            if isinstance(first, dict):
                try:
                    x = torch.utils.data._utils.collate.default_collate(x)  # dict[str, Tensor]
                except Exception:
                    # Minimal manual stack for common dict-of-tensors
                    keys = first.keys()
                    x = {k: torch.stack([sample[k] for sample in x]) for k in keys}
            elif isinstance(first, torch.Tensor):
                x = torch.stack(list(x))
            else:
                raise TypeError(f"Unsupported list element type for model forward: {type(first)}")

        if isinstance(x, dict):
            # Remove non-model keys often attached by dataloaders
            # (e.g., labels for supervision that HF models like CLIP do not accept)
            x = {k: v for k, v in x.items() if k not in ("label", "labels")}

            # move to device and cast only floating tensors to desired dtype
            x = {
                k: (
                    v.to(self.device, dtype=self.dtype)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                )
                for k, v in x.items()
            }
            try:
                sig = inspect.signature(self.model.forward)
                valid = set(sig.parameters.keys())
                filtered = {k: v for k, v in x.items() if k in valid}
                if len(filtered) == 0:
                    filtered = x
                with cm:
                    return self.model(**filtered)
            except Exception:
                with cm:
                    return self.model(**x)
        elif isinstance(x, torch.Tensor):
            # Cast floats, keep ints as-is
            x = x.to(self.device, dtype=self.dtype) if x.is_floating_point() else x.to(self.device)
            # Try to map raw tensor to a named kwarg (e.g., pixel_values) if present
            if x.dim() == 3:
                return x.squeeze(0)
            try:
                sig = inspect.signature(self.model.forward)
                params = sig.parameters
                preferred_order = [
                    "pixel_values",
                    "input_values",  # audio
                    "input_features",  # whisper-like
                    "input_ids",  # text
                ]
                for name in preferred_order:
                    if name in params:
                        with cm:
                            return self.model(**{name: x})
            except Exception:
                pass
            # Fallback: positional
            with cm:
                return self.model(x)
        else:
            raise TypeError(f"Unsupported input type for model forward: {type(x)}")

    def _consume_cache(self) -> torch.Tensor:
        if self._cache is None:
            return None
            # raise RuntimeError("Hook cache is empty. Did you call forward(return_hooked_out=True)?")
        z = self._cache
        self._cache = None
        # Unpack when modules return tuple/list; take the first tensor-like item
        if isinstance(z, (tuple, list)):
            selected = None
            for item in z:
                if isinstance(item, torch.Tensor):
                    selected = item
                    break
            if selected is None:
                raise TypeError("Hook output tuple/list contains no tensor; customize hook to select target output.")
            z = selected
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Hook output is not a tensor: {type(z)}")
        return z

    def _resolve_hook_target(self, root: torch.nn.Module, path: str) -> torch.nn.Module:
        """
        Resolve a dotted module path; supports list/Sequential indices.
        Example paths:
          "vision_model.encoder.layers" + hook_layer=10
          "encoder.layer.11.output"
          "resid" (if you alias it yourself upstream)
        """
        cur: Any = root
        for token in path.split("."):
            if token.isdigit():
                cur = cur[int(token)]
            else:
                if not hasattr(cur, token):
                    raise AttributeError(f"Module has no attribute '{token}' in path '{path}'")
                cur = getattr(cur, token)

        if not isinstance(cur, torch.nn.Module):
            raise TypeError(f"Resolved hook target '{path}' is not an nn.Module")
        return cur

    @contextlib.contextmanager
    def _hooked_model(self, replace_with: Optional[torch.Tensor], stop_after: bool = False):
        """
        Context manager:
        - If replace_with is None: capture the hook point output to self._cache.
        - Else: replace the hook point output with replace_with (shape must match).
        """
        handle = None

        def hook_fn(_module, _inputs, output):
            if replace_with is None:
                # capture path
                self._cache = output
                if stop_after:
                    # Abort the remainder of the forward pass immediately
                    raise _EarlyStopForward()
                return output
            else:
                # replace path
                if isinstance(output, (tuple, list)):
                    # Some modules return tuples (e.g., (hidden_states, other))
                    # We replace the first tensor-like element; adapt if needed.
                    idx0 = 0
                    orig = output[idx0]
                    if not isinstance(orig, torch.Tensor):
                        raise TypeError("Cannot replace non-tensor output; customize hook_fn for your module.")
                    self._assert_compatible(orig, replace_with)
                    return type(output)([replace_with] + list(output)[1:])
                else:
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("Cannot replace non-tensor output; customize hook_fn for your module.")
                    self._assert_compatible(output, replace_with)
                    return replace_with

        try:
            handle = self._hook_target.register_forward_hook(hook_fn)
            yield
        finally:
            if handle is not None:
                handle.remove()

    @staticmethod
    def _assert_compatible(ref: torch.Tensor, other: torch.Tensor):
        if ref.shape != other.shape:
            raise ValueError(f"Hook replacement shape mismatch: ref {tuple(ref.shape)} vs other {tuple(other.shape)}")
        if ref.dtype != other.dtype:
            # allow mild dtype diffs if you want; here we’re strict
            raise ValueError(f"Hook replacement dtype mismatch: ref {ref.dtype} vs other {other.dtype}")
        if ref.device != other.device:
            raise ValueError(f"Hook replacement device mismatch: ref {ref.device} vs other {other.device}")

    # ---------- Utilities ----------
    def _make_dummy_input(self, batch_size: int = 1) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Try to construct a model-appropriate dummy input:
        - If forward has `pixel_values`, create an image batch using config sizes.
        - If forward has `input_ids`, create random token ids and attention mask.
        - Else, fall back to a standard vision-shaped tensor using best-effort config.
        """

        try:
            sig = inspect.signature(self.model.forward)
            params = sig.parameters
        except Exception:
            params = {}

        cfg = getattr(self.model, "config", None)
        vision_cfg = getattr(cfg, "vision_config", None) if cfg is not None else None
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None

        def get_image_size_and_channels():
            # resolve size
            size = 224
            channels = 3
            if vision_cfg is not None:
                size = getattr(vision_cfg, "image_size", size)
                channels = getattr(vision_cfg, "num_channels", channels)
            elif cfg is not None:
                size = getattr(cfg, "image_size", size)
                channels = getattr(cfg, "num_channels", channels)
            return int(size), int(channels)

        # Build composite inputs if needed (e.g., CLIP expects both image and text)
        inputs: Dict[str, torch.Tensor] = {}

        if "pixel_values" in params:
            size, channels = get_image_size_and_channels()
            inputs["pixel_values"] = torch.randn(batch_size, channels, size, size, device=self.device, dtype=self.dtype)

        if "input_ids" in params:
            vocab_size = 30522
            seq_len = 16
            src_cfg = text_cfg if text_cfg is not None else cfg
            if src_cfg is not None:
                vocab_size = int(
                    getattr(
                        src_cfg,
                        "vocab_size",
                        getattr(src_cfg, "text_vocab_size", vocab_size),
                    )
                )
                seq_len = int(
                    getattr(
                        src_cfg,
                        "max_position_embeddings",
                        getattr(src_cfg, "context_length", seq_len),
                    )
                )
            inputs["input_ids"] = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            if "attention_mask" in params:
                inputs["attention_mask"] = torch.ones(batch_size, seq_len, device=self.device, dtype=torch.long)

        if len(inputs) > 0:
            return inputs

        # 3) Fallback: assume vision tensor input
        size, channels = get_image_size_and_channels()
        return torch.randn(batch_size, channels, size, size, device=self.device, dtype=self.dtype)
