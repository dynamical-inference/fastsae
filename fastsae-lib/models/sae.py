import dataclasses
import random
from typing import Any, Dict, Optional, Tuple

from config_dataclass import config_field
from config_dataclass import ConfigurableTorchModule
from config_dataclass import torch_dataclass
import einops
from fastsae.utils import verbose
import numpy as np
import torch
from torch import nn
from torch import Tensor


@torch_dataclass
class VanillaSAE(ConfigurableTorchModule, nn.Module):
    d_in: int = config_field(default=None)
    d_sae_factor: int = config_field(default=None)
    architecture: str = config_field(default="vanilla")
    activation_name: str = config_field(default="relu")
    device: str = config_field(default="cuda")
    seed: Optional[int] = config_field(default=None)
    _verbose: bool = dataclasses.field(default_factory=verbose.verbose_factory)

    def __lazy_post_init__(self):
        self.d_sae = self.d_in * self.d_sae_factor
        self.dtype = torch.float32  # TODO: what is the best way to set dtype?

        if self._verbose:
            print("🚀 Building SAE (lazy initialization)...")

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            print(f"🚨 Seed has been set to {self.seed}")

        # initialize encoder and decoder weightsv
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.d_in, self.d_sae, dtype=self.dtype)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().transpose(0, 1))

        self.b_enc = nn.Parameter(torch.zeros(self.d_sae, dtype=self.dtype))
        # decoder bias will be reinitialized with geometric median of activations later
        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype))

        def _get_activation_fn(name: str):
            n = name.lower()
            if n == "relu":
                return nn.ReLU()
            if n == "gelu":
                return nn.GELU()
            if n == "softplus":
                return nn.Softplus()
            raise ValueError(f"Unknown activation: {name}")

        self.activation = _get_activation_fn(self.activation_name)

        if self._verbose:
            print("✅ SAE built successfully!")

        self.to(self.device)

    @torch.no_grad()
    def _get_init_b_dec(self, z, init_method, maxiter=100):
        assert init_method == "geometric_median"
        from geom_median.torch import compute_geometric_median

        prev_b_dec = self.b_dec.clone().cpu()
        z = z.cpu()
        median = compute_geometric_median(z, skip_typechecks=True, maxiter=maxiter, per_component=False).median

        median = median.mean(dim=0) if len(median.shape) == 2 else median

        prev_dist = torch.norm(z - prev_b_dec, dim=-1)
        new_dist = torch.norm(z - median, dim=-1)

        if self._verbose:
            print(f"Previous distances: {prev_dist.median(0).values.mean().item()}")
            print(f"New distances: {new_dist.median(0).values.mean().item()}")

        new_b_dec = median.clone().detach().to(dtype=self.dtype, device=self.device)
        return new_b_dec

    @torch.no_grad()
    def init_b_dec(self, z, init_method):
        if self._verbose:
            print("🥐 Initializing b_dec ...")
            print(f"Init method: {init_method}")

        init_value = self._get_init_b_dec(z, init_method)
        self.b_dec.data = init_value

        if self._verbose:
            print("✅ b_dec initialized successfully!")

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def encode(self, z):
        a_pre = self._encode(z)
        a = self.activation(a_pre)
        return a

    def _encode(self, z):
        # centralize
        z_centralized = z - self.b_dec
        # encoder
        a_pre = einops.einsum(z_centralized, self.W_enc, "... d_in, d_in d_sae -> ... d_sae") + self.b_enc
        return a_pre

    def decode(self, a):
        # decoder
        z_hat = einops.einsum(a, self.W_dec, "... d_sae, d_sae d_in -> ... d_in") + self.b_dec
        return z_hat

    def steer(self, a: Tensor, steer_cfg: Optional[Dict[str, Any]]) -> Tensor:
        """
        Steering supports:
          steer_cfg = {
            "indices": int | Iterable[int] | Callable[[Tensor], Tensor],
                # which latents to affect; can be:
                #   - int → single index
                #   - iterable of ints → fixed indices
                #   - callable: (a: Tensor) -> LongTensor[batch, k] or [batch]
                #       e.g. lambda x: torch.topk(x, k=3, dim=-1).indices
            # exactly one of the following operations:
            "multiply": float,                  # a[idx] *= multiply
            "add": float,                       # a[idx] += add
            "set": float,                       # a[idx] = set
            "permute": bool,                    # randomly permute activations at given indices
          }
        """
        if not steer_cfg or "indices" not in steer_cfg:
            return a

        idx = steer_cfg["indices"]
        if callable(idx):
            idx = idx(a)
        else:
            if isinstance(idx, int):
                idx = [idx]
            idx = torch.as_tensor(idx, device=a.device)

        # Broadcast-friendly index
        # a shape: [..., d_sae]; we update last dim at given indices
        op_count = sum(k in steer_cfg for k in ("multiply", "add", "set", "permute"))
        if op_count != 1:
            raise ValueError("Provide exactly one of {'multiply','add','set'} in steer_cfg.")

        if idx.ndim == 1:
            # same indices for all samples
            if "set" in steer_cfg:
                a[..., idx] = float(steer_cfg["set"])
            elif "multiply" in steer_cfg:
                a[..., idx] = a[..., idx] * float(steer_cfg["multiply"])
            elif "add" in steer_cfg:
                a[..., idx] = a[..., idx] + float(steer_cfg["add"])
            elif "permute" in steer_cfg:
                perm = idx[torch.randperm(len(idx), device=a.device)]
                a[..., idx] = a[..., perm]
        else:
            # per-sample indices: idx shape [batch, k]
            d_batch = a.shape[0]
            for b in range(d_batch):
                if "set" in steer_cfg:
                    a[b, ..., idx[b]] = float(steer_cfg["set"])
                elif "multiply" in steer_cfg:
                    a[b, ..., idx[b]] = a[b, ..., idx[b]] * float(steer_cfg["multiply"])
                elif "add" in steer_cfg:
                    a[b, ..., idx[b]] = a[b, ..., idx[b]] + float(steer_cfg["add"])
                elif "permute" in steer_cfg:
                    perm = idx[b][torch.randperm(idx.shape[1], device=a.device)]
                    a[b, ..., idx[b]] = a[b, ..., perm]

        return a

    def forward(
        self,
        z: Tensor,
        steer_cfg: Optional[Dict[str, Any]] = None,
        return_act: bool = True,
    ) -> Tuple[Tensor, Tensor] | Tensor:
        """
        z -> a -> (optional steering) -> z_hat
        """
        a = self.encode(z)
        if steer_cfg is not None:
            a = self.steer(a, steer_cfg)
        z_hat = self.decode(a)
        return (z_hat, a) if return_act else z_hat

    # Optional helper for training losses
    def aux_losses(self, a: Tensor) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        if self.cfg.l1_coeff > 0:
            losses["l1"] = self.cfg.l1_coeff * a.abs().mean()
        return losses
