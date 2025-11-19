import dataclasses
import functools
from pathlib import Path
from typing import Any

import config_dataclass
from fastsae.models.backbone import Backbone
from fastsae.models.sae import VanillaSAE
from fastsae.utils import verbose
import torch
import torch.optim
import torch.utils.data


def clone_dataloader(dataloader: torch.utils.data.DataLoader, new_dataset) -> torch.utils.data.DataLoader:
    """
    Creates a new DataLoader with the same parameters as the original one, but with a new dataset.
    This function is necessary because we cannot simply replace the .dataset attribute of a DataLoader,
    as the sampler is tied to the old dataset.

    Note: This function may not work correctly with custom samplers that cannot be re-initialized
    with a new dataset. It is designed to work with standard PyTorch samplers.
    """
    # These arguments are mutually exclusive and handled by the dataloader.
    # We can infer shuffle from the sampler type.
    shuffle = isinstance(dataloader.sampler, torch.utils.data.RandomSampler)

    # If a custom sampler is used, we cannot reliably re-initialize it.
    # We only handle standard samplers here.
    if not isinstance(
        dataloader.sampler,
        (torch.utils.data.RandomSampler, torch.utils.data.SequentialSampler),
    ):
        raise ValueError("Cannot clone DataLoader with a custom sampler.")

    # We also don't handle custom batch_samplers.
    if not isinstance(dataloader.batch_sampler, torch.utils.data.BatchSampler):
        raise ValueError("Cannot clone DataLoader with a custom batch_sampler.")

    return torch.utils.data.DataLoader(
        new_dataset,
        batch_size=dataloader.batch_size,
        shuffle=shuffle,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        generator=dataloader.generator,
    )


cache_dir = Path("data/cache")
cache_dir.mkdir(parents=True, exist_ok=True)


def backbone_encode(backbone, x, indices=None):
    preprocessed_data = dict(latent=backbone.forward_to_z(x))
    if indices is not None:
        preprocessed_data["indices"] = indices
    return preprocessed_data


def backbone_encode_nograd(backbone, x, indices=None):
    with torch.no_grad():
        return backbone_encode(backbone, x, indices)


def get_cached_dataloader(
    dataloader,
    backbone,
    batch_size=128,
    writer_batch_size=128,
):
    img_dataset = dataloader.dataset

    pre_processed_name = f"cached_latents/{img_dataset._fingerprint}/{backbone.model_hf_name}/{backbone.hook_target_name}"

    compute_latents = functools.partial(backbone_encode_nograd, backbone)
    preprocessed_dataset = img_dataset.map(
        compute_latents,
        with_indices=True,
        batched=True,
        keep_in_memory=False,
        batch_size=batch_size,
        writer_batch_size=writer_batch_size,
        desc=f"Precomputing {pre_processed_name}...",
        remove_columns=img_dataset.column_names,
    )
    preprocessed_dataset.reset_format()
    preprocessed_dataset = preprocessed_dataset.with_format("torch")

    return clone_dataloader(dataloader, preprocessed_dataset)


@config_dataclass.torch_dataclass
class SAEWrapper(config_dataclass.ConfigurableTorchModule, torch.nn.Module):
    """
    Wraps:
      - Backbone (HF-based or local model; supports hooking a mid-layer and replacing it)
      - SAE (reconstruct/steer hooked features)
    Forward contract:
      forward(x, steer_cfg=None) -> (out_pre, out_post, a)
        1) Run backbone to capture z at hook point → out_pre, z
        2) SAE(z, steer_cfg) → z_hat, activations a
        3) Re-run backbone replacing hook output with z_hat → out_post
    """

    backbone: Backbone
    sae: VanillaSAE
    _verbose: bool = dataclasses.field(default_factory=verbose.verbose_factory)
    _caching: bool = dataclasses.field(default=True)
    _caching_batch_size: int = dataclasses.field(default=2**11)
    _caching_writer_batch_size: int = dataclasses.field(default=2**11)

    def __lazy_post_init__(self):
        if self._verbose:
            print("🚀 Building Wrapper (lazy initialization)...")

        assert self.backbone.device == self.sae.device
        assert self.backbone.dtype == self.sae.dtype

        self.device = self.backbone.device
        self.dtype = self.backbone.dtype

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        self.sae = self.sae.to(self.device)
        self.sae.eval()

        if self._verbose:
            print("✅ Wrapper built successfully!")

    # ---------- I/O: save / load ----------

    def save(self, path):
        self.backbone.save(path)
        self.sae.save(path)

    @classmethod
    def load(cls, path):
        backbone = Backbone.load(path)
        sae = VanillaSAE.load(path)
        return cls(backbone=backbone, sae=sae)

    # ---------- Core API ----------

    def forward(self, x, steer_cfg=None):
        out_pre, z = self.backbone(x, return_hooked_out=True)
        z_hat, a = self.sae(z, steer_cfg)
        out_post = self.backbone.forward_steered(x, z_hat)
        return out_pre, out_post, a

    def encode(self, x, steer_cfg=None):
        z = backbone_encode(self.backbone, x)
        _, a = self.sae(z, steer_cfg)
        return a

    # ---------- Utilities ----------

    def to(self, device: torch.device | str):  # type: ignore[override]
        device = torch.device(device)
        self.device = device
        self.backbone.to(device)
        self.sae.to(device)
        return self

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        total_n_training_samples: int,
        sparsity_coefficient: float,
        **kwargs,
    ):
        # Import SAETrainer here to avoid circular imports
        from fastsae.training.trainer import SAETrainer

        if self._caching:
            print("Preparing cached dataloader...")
            dataloader = get_cached_dataloader(
                dataloader,
                backbone=self.backbone,
                batch_size=self._caching_batch_size,
                writer_batch_size=self._caching_writer_batch_size,
            )

        trainer = SAETrainer(
            wrapped=self,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            total_n_training_samples=total_n_training_samples,
            sparsity_coefficient=sparsity_coefficient,
            **kwargs,
        )

        return trainer.train()

    def eval(self):
        self.backbone.eval()
        self.sae.eval()
        return super().eval()
