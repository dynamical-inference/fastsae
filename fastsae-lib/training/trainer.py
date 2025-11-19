import dataclasses
import pathlib
from typing import Any, NamedTuple, Optional

from config_dataclass import config_dataclass
from config_dataclass import config_field
from config_dataclass import Configurable
from fastsae.models.wrapper import SAEWrapper
from fastsae.utils import verbose
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


class LossOutput(NamedTuple):
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    auxiliary_loss: torch.Tensor


class MetricsOutput(NamedTuple):
    """
    l0: the avg number of latents firing per input activation, to evaluate sparsity
    explained_variance: MSE relative to predicting the mean activation of the batch, to evaluate reconstruction quality
    """

    l0: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_std: torch.Tensor


@config_dataclass
class SAETrainer(Configurable):
    wrapped: SAEWrapper
    dataloader: DataLoader
    optimizer: Optimizer
    scheduler: Any

    sae_b_dec_init_method: str = config_field(default="geometric_median")

    total_n_training_samples: int = config_field()
    total_n_checkpoints: int = config_field(default=10)

    recon_coefficient: float = config_field(default=1.0)
    sparsity_coefficient: float = config_field()
    auxiliary_coefficient: float = config_field(default=1.0)

    checkpoint_path: Optional[str] = config_field(default=None)

    log_to_wandb: bool = config_field(default=False)
    wandb_log_frequency: int = config_field(default=10)
    wandb_project: Optional[str] = config_field(default=None)
    wandb_entity: Optional[str] = config_field(default=None)
    wandb_run_id: Optional[str] = config_field(default=None)

    use_ghost_grads: bool = config_field(default=True)
    latent_sampling_method: str = config_field(default=None)
    latent_sampling_window: int = config_field(default=64)
    resample_batches: int = config_field(default=32)
    latent_reinit_scale: float = config_field(default=0.2)
    dead_latent_window: int = config_field(default=64)
    dead_latent_estimation_method: str = config_field(default="no_fire")
    dead_latent_threshold: float = config_field(default=1e-6)

    _verbose: bool = dataclasses.field(default_factory=verbose.verbose_factory)

    def __post_init__(self):
        self.d_sae = self.wrapped.sae.d_sae
        self.device = self.wrapped.device
        self.dtype = self.wrapped.dtype

        self._n_training_samples = 0  # number of samples seen so far
        self._n_training_steps = 0  # number of training steps so far (n_seen_samples // batch_size)
        self._n_forward_passes_since_fired = torch.zeros(self.d_sae, device=self.device)
        self._n_frac_active_tokens = 0
        self._ghost_grad_neuron_mask = None
        self._a_freq_scores = torch.zeros(self.d_sae, device=self.device)

        self._checkpoint_thresholds = list(
            range(
                0,
                self.total_n_training_samples,
                self.total_n_training_samples // self.total_n_checkpoints,
            )
        )[1:]

        if self.log_to_wandb:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=self.to_dict(),
                id=self.wandb_run_id,
            )

    # === Loss functions ===

    def _mse_ghost_loss(self, z, z_hat, a, z_mse_loss):
        self._ghost_grad_neuron_mask = dead_mask = (self._n_forward_passes_since_fired > self.dead_latent_window).bool()

        loss = torch.tensor(0.0, dtype=self.wrapped.dtype, device=self.wrapped.device)

        if dead_mask is not None and dead_mask.sum() > 0:
            # 1.
            z_diff = (z - z_hat).detach().float()
            z_diff_l2_norm = torch.norm(z_diff, dim=-1)

            # 2. Apply mask on the last dimension while preserving original shape
            dead_mask_broadcast = dead_mask.view(*([1] * (a.ndim - 1)), -1)
            dead_acts = torch.exp(a * dead_mask_broadcast)

            # Equivalent to selecting only dead latents and multiplying by W_dec[dead_mask]
            # because (A S)(S^T W) == A (S S^T) W where S selects dead columns
            ghost_z_hat = dead_acts @ self.wrapped.sae.W_dec
            ghost_z_hat_l2_norm = torch.norm(ghost_z_hat, dim=-1)

            # 3.
            scaling_factor = z_diff_l2_norm / (1e-6 + ghost_z_hat_l2_norm * 2)
            ghost_z_hat = ghost_z_hat * scaling_factor[..., None].detach()

            # 4.
            loss = torch.pow((ghost_z_hat - z_diff), 2) / (z_diff**2).sum(dim=-1, keepdim=True).sqrt()
            scaling_factor = (z_mse_loss / (loss + 1e-6)).detach()
            loss = loss * scaling_factor

        return loss

    def loss_fn(self, z, a, z_hat):
        z_mse_loss = torch.pow((z_hat - z.float()), 2) / (z**2).sum(dim=-1, keepdim=True).sqrt()
        a_l1_loss = torch.abs(a).sum(dim=-1).mean(dim=(0,))
        z_mse_ghost_loss = self._mse_ghost_loss(z, z_hat, a, z_mse_loss)

        loss_dict = {
            "recon_loss": z_mse_loss.mean(),
            "sparsity_loss": a_l1_loss.mean(),
            "auxiliary_loss": z_mse_ghost_loss.mean(),
        }

        return LossOutput(**loss_dict)

    def metric_fn(self, z, a, z_hat):
        a_l0 = (a > 0).float().sum(-1).mean()
        # Per-dimension MSE and variance
        z_mse = ((z_hat - z) ** 2).mean(dim=0)
        total_variance = z.var(dim=0, unbiased=False)
        explained_variance = 1 - z_mse / total_variance.clamp_min(1e-12)  # avoid div/0

        metrics_dict = {
            "l0": a_l0,
            "explained_variance": explained_variance.mean(),
            "explained_variance_std": explained_variance.std(),
        }

        return MetricsOutput(**metrics_dict)

    def compute_total_loss(self, loss_outputs):
        weighted_sum = (
            self.recon_coefficient * loss_outputs.recon_loss
            + self.sparsity_coefficient * loss_outputs.sparsity_loss
            + self.auxiliary_coefficient * loss_outputs.auxiliary_loss
        )
        return weighted_sum.mean()

    # === Logging ===

    def _log_wandb_if_needed(self, loss_outputs, metrics_outputs, weighted_loss_sum):
        if self.log_to_wandb and (self._n_training_steps + 1) % self.wandb_log_frequency == 0:
            a_freq = self._a_freq_scores / self._n_frac_active_tokens
            """Log training metrics to wandb."""
            log_dict = {
                "metrics/l0": metrics_outputs.l0.item(),
                "metrics/explained_variance": metrics_outputs.explained_variance.item(),
                "metrics/explained_variance_std": metrics_outputs.explained_variance_std.item(),
                "sparsity/mean_passes_since_fired": self._n_forward_passes_since_fired.mean().item(),
                "sparsity/dead_latents": (a_freq < self.dead_latent_threshold).float().mean().item(),
                "details/n_training_samples": self._n_training_samples,
                "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            overall_loss = 0.0
            for k, v in loss_outputs._asdict().items():
                log_dict[f"losses/{k}"] = v.item()
                overall_loss += v

            log_dict["losses/weighted_loss_sum"] = weighted_loss_sum.item()
            log_dict["losses/overall_loss"] = overall_loss.item()

            if self._ghost_grad_neuron_mask is not None:
                log_dict["sparsity/n_passes_since_fired_over_threshold"] = self._ghost_grad_neuron_mask.sum().item()

            wandb.log(log_dict, step=self._n_training_steps)

    @torch.no_grad()
    def _update_pbar(self, loss_outputs, pbar, batch_size):
        loss_str = " | ".join([f"{k}: {v.item():.3f}" for k, v in loss_outputs._asdict().items()])

        pbar.set_description(f"{self._n_training_steps}| {loss_str}")
        pbar.update(batch_size)

    # === Checkpointing ===

    def save_backbone(self):
        save_path = pathlib.Path(self.checkpoint_path) / "backbone"
        save_path.mkdir(parents=True, exist_ok=True)
        self.wrapped.backbone.save(save_path)

        if self._verbose:
            print(f"📌 Saved Backbone to {save_path}")

    def _save_sae_if_needed(self):
        if self._checkpoint_thresholds and self._n_training_samples > self._checkpoint_thresholds[0]:
            save_path = f"{self.checkpoint_path}/{self._n_training_samples}"
            self.wrapped.sae.save(save_path)

            if self._verbose:
                print(f"📌 Saved SAE to {save_path}")

            self._checkpoint_thresholds.pop(0)

    def save_sae(self, is_final=False):
        save_path = f"{self.checkpoint_path}"
        if is_final:
            save_path = f"{save_path}/final"
        self.wrapped.sae.save(save_path)

        if self._verbose:
            print(f"📌 Saved SAE to {save_path}")

    # === Resetting running sparsity stats ===

    def _reset_running_sparsity_stats_if_needed(self):
        if (self._n_training_steps + 1) % self.latent_sampling_window == 0:
            self._a_freq_scores = torch.zeros(self.d_sae, device=self.device)
            self._n_frac_active_tokens = 0

    # === Training ===
    def initialize_b_dec(self):
        for batch in self.dataloader:
            break
        z = self.wrapped.backbone.forward_to_z(batch)

        # TODO: currently only supports geometric median
        self.wrapped.sae.init_b_dec(z, self.sae_b_dec_init_method)

    # on-the-fly training
    def train(self):
        # for efficient matrix multiplication
        torch.set_float32_matmul_precision("high")

        self.initialize_b_dec()

        if self.checkpoint_path is not None:
            self.save_backbone()

        pbar = tqdm(total=self.total_n_training_samples, desc="Training SAE")
        try:
            dataloader = self.dataloader
            while self._n_training_samples < self.total_n_training_samples:
                for backbone_inputs in dataloader:
                    if self._n_training_samples >= self.total_n_training_samples:
                        break

                    self._reset_running_sparsity_stats_if_needed()

                    # set decoder norm to unit norm BEFORE forward pass
                    self.wrapped.sae.set_decoder_norm_to_unit_norm()  # TODO: let this be configurable

                    # forward pass
                    z = self.wrapped.backbone.forward_to_z(backbone_inputs)
                    z_hat, a = self.wrapped.sae(z)

                    # compute loss
                    loss_outputs = self.loss_fn(z, a, z_hat)
                    loss = self.compute_total_loss(loss_outputs)
                    metrics_outputs = self.metric_fn(z, a, z_hat)

                    # for auxiliary loss (ghost gradient)
                    # Handle both 2D (batch_size, sae_dim) and 3D (batch_size, seq_len, sae_dim) tensors
                    if a.dim() == 2:
                        # 2D case: (batch_size, sae_dim)
                        # did_fire should indicate which SAE latents fired (across the batch)
                        did_fire = (a > 0).float().sum(0) > 0  # (sae_dim,) boolean - which latents fired
                        self._a_freq_scores += (a.abs() > 0).float().sum(0)  # (sae_dim,)
                    else:
                        # 3D+ case: (batch_size, seq_len, sae_dim) or higher
                        did_fire = (((a > 0).float().sum(-2) > 0).sum(-2)) > 0
                        self._a_freq_scores += (a.abs() > 0).float().sum(0).sum(0)

                    self._n_forward_passes_since_fired += 1
                    self._n_forward_passes_since_fired[did_fire] = 0
                    self._n_frac_active_tokens += z.size(0)

                    # backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.wrapped.sae.remove_gradient_parallel_to_decoder_directions()
                    self.optimizer.step()
                    self.scheduler.step()

                    # update training stats, pbar, log, and save
                    self._n_training_samples += z.size(0)
                    self._n_training_steps += 1

                    self._update_pbar(loss_outputs, pbar, z.size(0))
                    self._log_wandb_if_needed(loss_outputs, metrics_outputs, loss)
                    if self.checkpoint_path is not None:
                        self._save_sae_if_needed()
        finally:
            if self.checkpoint_path is not None:
                self.save_sae(is_final=True)

        pbar.close()
        return self.wrapped.sae

    def run_eval(self):
        # TODO: Implement this
        pass
