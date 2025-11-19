from collections import defaultdict
from collections.abc import Mapping
import dataclasses
import json
import os
import pickle
from typing import Any

from config_dataclass import config_dataclass
from config_dataclass import config_field
from config_dataclass import Configurable
from fastsae.models.wrapper import SAEWrapper
from fastsae.utils import verbose
from fastsae.utils.run_utils import get_len_dataset
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@config_dataclass
class SAEInterpreter(Configurable):
    wrapped: SAEWrapper
    backbone_processor: Any
    stats_dataloader: DataLoader
    stats_labels_np: np.array = None
    stats_top_k_max_samples: int = config_field(default=20)
    sparsity_thr: list[float] = config_field(default_factory=list)
    stats: dict = config_field(default_factory=dict)
    _verbose: bool = dataclasses.field(default_factory=verbose.verbose_factory)

    def __post_init__(self):
        self.stats = {}
        # self.stats_dataset = self.stats_dataloader.dataset.base_dataset
        self.stats_dataset = self.stats_dataloader.dataset
        self.stats_labels_np = self.stats_labels_np

    def _r2_score(self, x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/metrics.py#L316
        Compute the R^2 score (coefficient of determination) for the reconstruction.
        A score of 1 indicates a perfect reconstruction while a score of 0 indicates
        that the reconstruction is as good as the mean.

        """
        assert x.shape == x_hat.shape, "Input tensors must have the same shape"
        assert len(x.shape) == 2, "Input tensors must be 2D"

        ss_res = torch.mean((x - x_hat) ** 2)
        ss_tot = torch.mean((x - x.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + eps))
        return r2

    def compute_stats(self, compute_cls_wise_top_idx: bool = True):
        n_iter_samples = 0
        torch.set_float32_matmul_precision("high")

        stats = {
            "mean_act": torch.zeros([self.wrapped.sae.d_sae]),
            "sparsity": torch.zeros([self.wrapped.sae.d_sae]),
            "sparsity_thr": {thr: torch.zeros([self.wrapped.sae.d_sae]) for thr in self.sparsity_thr},
            "top_act": torch.zeros([self.stats_top_k_max_samples, self.wrapped.sae.d_sae]),
            "top_idx": torch.zeros([self.stats_top_k_max_samples, self.wrapped.sae.d_sae]),
            "l2_loss": 0.0,
            "r2_score": 0.0,
        }

        total_iter_samples = get_len_dataset(self.stats_dataset)  # TODO: check if this is correct
        labels_np = self.stats_labels_np

        if compute_cls_wise_top_idx:
            cls_wise_sae_acts = defaultdict(lambda: torch.zeros(self.wrapped.sae.d_sae))

        n_batch = 0
        pbar = tqdm(total=total_iter_samples, desc="Computing SAE statistics")
        try:
            dataloader_iter = iter(self.stats_dataloader)
            while n_iter_samples < total_iter_samples:
                try:
                    backbone_inputs = next(dataloader_iter)
                except StopIteration:
                    break

                with torch.no_grad():
                    z = self.wrapped.backbone.forward_to_z(backbone_inputs)
                    z_hat, a = self.wrapped.sae(z)
                    a_avg = a.mean(1).cpu()  # (batch_size, sae_dim)

                    z_flattened = z.reshape(-1, z.shape[-1])
                    z_hat_flattened = z_hat.reshape(-1, z_hat.shape[-1])
                    r2_val = self._r2_score(z_flattened, z_hat_flattened).item()
                    stats["r2_score"] += r2_val
                    n_batch += 1

                    z_mse_loss = nn.functional.mse_loss(z_hat, z, reduction="none")
                    z_mse_loss = z_mse_loss.mean(dim=(1, 2)).sum().item()  # per sample MSE, summed over batch
                    stats["l2_loss"] += z_mse_loss

                batch_size = a_avg.size(0)
                stats["mean_act"] += a_avg.sum(dim=0)  # sum over images; (sae_dim, )
                stats["sparsity"] += (a_avg > 0).sum(dim=0)  # sum over images; (sae_dim, )
                for thr in self.sparsity_thr:
                    stats["sparsity_thr"][thr] += (a_avg > thr).sum(dim=0)

                if compute_cls_wise_top_idx:
                    labels = labels_np[n_iter_samples : n_iter_samples + batch_size]
                    for i, label in enumerate(labels):
                        cls_wise_sae_acts[label] += a_avg[i]

                top_k = min(self.stats_top_k_max_samples, a_avg.size(0))
                batch_act, batch_idx = torch.topk(a_avg, k=top_k, dim=0)  # batch_act (top_k, sae_dim)
                batch_idx += n_iter_samples

                total_act = torch.cat([stats["top_act"], batch_act], dim=0)
                total_idx = torch.cat([stats["top_idx"], batch_idx], dim=0)

                top_act, idx_of_idx = torch.topk(total_act, k=self.stats_top_k_max_samples, dim=0)
                top_idx = torch.gather(total_idx, 0, idx_of_idx)

                stats["top_act"] = top_act
                stats["top_idx"] = top_idx

                n_iter_samples += batch_size

                pbar.set_description(f"[{n_iter_samples}/{total_iter_samples}] images processed")
                pbar.update(batch_size)

                del z, z_hat, a, a_avg
        finally:
            stats["mean_act"] /= stats["sparsity"].clamp(min=1)
            stats["sparsity"] /= n_iter_samples
            stats["top_act"] = stats["top_act"].transpose(0, 1)
            stats["top_idx"] = stats["top_idx"].transpose(0, 1)
            stats["l2_loss"] /= n_iter_samples
            stats["r2_score"] /= n_batch
            stats["l0_0"] = stats["sparsity"].mean().item()

            for thr in self.sparsity_thr:
                stats["sparsity_thr"][thr] /= n_iter_samples
                stats[f"l0_{thr}"] = stats["sparsity_thr"][thr].mean().item()

        zero_mask = stats["top_act"] <= 0

        top_idx = stats["top_idx"].numpy().astype(int)
        labels_np = self.stats_labels_np

        top_label = labels_np[top_idx]
        top_label = torch.as_tensor(top_label)
        top_label[zero_mask] = -1

        stats["top_idx"] = stats["top_idx"].int()
        stats["top_label"] = top_label
        stats["top_entropy"] = self.calculate_entropy(stats["top_act"], stats["top_label"])

        stats["top_act"][zero_mask] = -1
        stats["top_idx"][zero_mask] = -1

        if compute_cls_wise_top_idx:
            stats["cls_wise_top_idx"] = {
                label: torch.topk(cls_wise_sae_acts[label], k=self.stats_top_k_max_samples, dim=0).indices for label in cls_wise_sae_acts.keys()
            }

        if self._verbose:
            print("✅ Stats computed successfully!")

        self.stats = stats

    def calculate_entropy(self, top_act, top_label, ignore_label_idx: int = None, eps=1e-9):
        dict_size = top_label.shape[0]
        entropy = torch.zeros(dict_size)

        top_label = torch.as_tensor(top_label)

        for i in range(dict_size):
            unique_labels, counts = top_label[i].unique(return_counts=True)
            if ignore_label_idx is not None:
                counts = counts[unique_labels != ignore_label_idx]
                unique_labels = unique_labels[unique_labels != ignore_label_idx]
            if len(unique_labels) != 0:
                if counts.sum().item() < 10:
                    entropy[i] = -1  # discount as too few datapoints!
                else:
                    summed_probs = torch.zeros_like(unique_labels, dtype=top_act.dtype)
                    for j, label in enumerate(unique_labels):
                        summed_probs[j] = top_act[i][top_label[i] == label].sum().item()
                    summed_probs = summed_probs / summed_probs.sum()
                    entropy[i] = -torch.sum(summed_probs * torch.log(summed_probs + eps))
            else:
                entropy[i] = -1
        return entropy

    def save_stats(self, path):
        out_dict = {}
        for key, value in self.stats.items():
            if isinstance(value, dict):
                # sparsity in different thresholds and cls-wise top idx
                with open(os.path.join(path, f"{key}.pkl"), "wb") as f:
                    pickle.dump(value, f)
                continue
            # Handle scalar-like values (float/int, 0-d torch tensors, 0-d/size-1 numpy arrays)
            if isinstance(value, (float, int, np.floating, np.integer)):
                out_dict[key] = round(float(value), 4)
                continue
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.numel() == 1:
                    out_dict[key] = round(value.detach().cpu().item(), 4)
                else:
                    torch.save(value.detach().cpu(), os.path.join(path, f"{key}.pt"))
                continue
            if isinstance(value, np.ndarray):
                if value.ndim == 0 or value.size == 1:
                    out_dict[key] = round(float(value), 4)
                else:
                    torch.save(torch.from_numpy(value), os.path.join(path, f"{key}.pt"))
                continue
            # Fallback: attempt to save; ignore if unsupported
            try:
                torch.save(value, os.path.join(path, f"{key}.pt"))
            except Exception:
                print(f"Warning: Failed to save {key}")
                torch.save(value.detach().cpu(), os.path.join(path, f"{key}.pt"))
                pass
        json.dump(out_dict, open(os.path.join(path, "metrics.json"), "w"), indent=4)

        if self._verbose:
            print("✅ Stats saved at", path)

    def load_stats(self, path):
        for file in os.listdir(path):
            if file.endswith(".pt"):
                self.stats[file.split(".")[0]] = torch.load(os.path.join(path, file), map_location="cpu").detach()
            elif file.endswith(".json"):
                with open(os.path.join(path, file), "r") as f:
                    self.stats[file.split(".")[0]] = json.load(f)
            elif file.endswith(".pkl"):
                with open(os.path.join(path, file), "rb") as f:
                    self.stats[file.split(".")[0]] = pickle.load(f)

            if file == "top_idx.pt":
                self.stats["top_idx"] = self.stats["top_idx"].int()

        if self._verbose:
            print("✅ Stats loaded from", path)

    # ---------- Visualization ----------
    def show_stats_scatter_plot(self, highlight_indices=None, mask=None, eps=1e-9, save_root=None):
        if mask is None:
            mask = torch.ones_like(self.stats["sparsity"], dtype=torch.bool)

        indices = torch.where(mask)[0]
        plotting_data = torch.stack(
            [
                torch.log10(self.stats["sparsity"][mask] + eps),
                torch.log10(self.stats["mean_act"][mask] + eps),
                self.stats["top_entropy"][mask],
                indices,
            ],
            dim=0,
        )
        plotting_data = plotting_data.transpose(0, 1)

        x_label = "log10(sparsity)"
        y_label = "log10(mean_act)"
        color_label = "entropy"
        hover_label = "index"

        df = pd.DataFrame(plotting_data.numpy(), columns=[x_label, y_label, color_label, hover_label])

        if highlight_indices is not None:
            fig = px.scatter(
                df,
                x=x_label,
                y=y_label,
                color=color_label,
                marginal_x="histogram",
                marginal_y="histogram",
                opacity=0.02,
                hover_data=[hover_label],
            )

            fig.add_trace(
                go.Scatter(
                    x=df.loc[highlight_indices, x_label],  # x values for the specific indices
                    y=df.loc[highlight_indices, y_label],  # y values for the specific indices
                    mode="markers",
                    marker=dict(color="red", size=10),  # Customize color and size
                    name="Highlighted Indices",
                    hoverinfo="text",
                    text=[
                        f"Index: {i},\nMean act: {df.loc[i, x_label]},\nSparsity: {df.loc[i, y_label]}" for i in highlight_indices
                    ],  # Add hover text for each point
                )
            )
        else:
            fig = px.scatter(
                df,
                x=x_label,
                y=y_label,
                color=color_label,
                marginal_x="histogram",
                marginal_y="histogram",
                opacity=0.5,
                hover_data=[hover_label],
            )

        if save_root is not None:
            fig.write_image(os.path.join(save_root, "scatter_plot.png"))
        fig.show()

    # ---------- Visualization (with inference) ----------
    def _prepare_inputs(self, images, do_normalize=True):
        inputs = self.backbone_processor(images=images, text="a photo", return_tensors="pt", do_normalize=do_normalize)
        if isinstance(inputs, Mapping):
            inputs = {k: inputs[k] for k in inputs.keys()}
        return inputs

    @torch.no_grad()
    def inference(self, x, steer_cfg=None):
        # if x is Image or a list of Images, convert to tensor
        if isinstance(x, Image.Image) or isinstance(x, list) and all(isinstance(img, Image.Image) for img in x):
            x = self._prepare_inputs(x)

        backbone = self.wrapped.backbone
        sae = self.wrapped.sae

        out_pre, z = backbone(x, return_hooked_out=True)
        z_hat, a = sae(z, steer_cfg)
        out_post = backbone.forward_steered(x, z_hat)

        z_recon_loss = (z_hat.float() - z.float()).pow(2).mean().item()
        sae_act_sparsity = (a == 0).float().mean().item()

        # TODO: this is for clip
        try:
            out_post = out_post.vision_model_output.last_hidden_state
            out_pre = out_pre.vision_model_output.last_hidden_state
        except AttributeError:
            out_post = out_post.last_hidden_state
            out_pre = out_pre.last_hidden_state
        output_recon_loss = (out_post.float() - out_pre.float()).pow(2).mean().item()

        return {
            "metrics/z_recon_loss": z_recon_loss,
            "metrics/sae_act_sparsity": sae_act_sparsity,
            "metrics/output_recon_loss": output_recon_loss,
            "sae_act": a,
        }

    def get_masked_image(self, images, latent_idx):
        batch_size = len(images)

        # inference
        inputs = self._prepare_inputs(images)  # need inputs for unfolding patches
        out_dicts = self.inference(inputs)

        # get seg mask
        act = out_dicts["sae_act"][:, 1:, latent_idx].cpu()  # TODO: automate to discard CLS (batch_size, num_patches,)
        act_image_sahpe = act.view(batch_size, 1, -1, 1, 1)
        seg_mask = (act_image_sahpe - act_image_sahpe.min()) / (act_image_sahpe.max() - act_image_sahpe.min() + 1e-10)
        seg_mask = seg_mask.clamp(0, 1)
        base_opacity = 0.1
        seg_mask = seg_mask * (1 - base_opacity) + base_opacity

        # convert image to patches
        inputs_unormalized = self._prepare_inputs(images, do_normalize=False)
        # TODO: where should I get image size?
        # image_size = self.wrapped.backbone.image_size
        image_size = inputs_unormalized["pixel_values"].shape[-1]
        patch_size = self.wrapped.backbone.patch_size
        num_patches = image_size // patch_size
        patches = inputs_unormalized["pixel_values"].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # torch.Size([B, 3, 16, 16, 14, 14])
        patches = patches.reshape(batch_size, 3, num_patches * num_patches, patch_size, patch_size)

        # apply seg mask
        weighted_patches = patches * seg_mask

        # reshape and permute
        output = weighted_patches.reshape(batch_size, 3, num_patches, num_patches, patch_size, patch_size)
        output = output.permute(0, 1, 2, 4, 3, 5)
        output = output.reshape(batch_size, 3, image_size, image_size)

        # Convert to plottable format [B, 224, 224, 3]
        return output.permute(0, 2, 3, 1)  # [B, 224, 224, 3]

    def show_sae_activation_bar(self, a):
        assert a.ndim == 1, "Activation values must be 1D"

        fig = go.Figure()
        entropy_values = self.stats["top_entropy"].numpy()

        # Add a trace for the line plot with different colors
        fig.add_trace(
            go.Scatter(
                y=a,
                mode="lines+markers",
                name="Activation Values",
                marker=dict(
                    color=entropy_values,
                    colorscale="thermal",
                    colorbar=dict(title="Top Entropy"),
                ),
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title="Activation Values Line Plot",
            xaxis_title="Index",
            yaxis_title="Activation Value",
            showlegend=False,
        )

        # Display the figure
        fig.show()

    def show_ref_samples(self, latent_idx: int, num_ref_samples: int = 5, input_img: Image.Image = None):
        description = f"{latent_idx}"
        resize_size = self.wrapped.backbone.image_size

        top_idx = self.stats["top_idx"][latent_idx][:num_ref_samples]
        valid_mask = top_idx != -1
        top_idx = top_idx[valid_mask]

        ref_imgs = []
        ref_labels = []

        for i, idx in enumerate(top_idx):
            img = self.stats_dataset[idx.item()]["image"]
            ref_imgs.append(img.resize((resize_size, resize_size)))
            assert self.stats_dataset[idx.item()]["label"] == self.stats["top_label"][latent_idx][i], "label mismatch, try matching dataset shuffle seed"
            ref_labels.append(self.stats_dataset[idx.item()]["label"])

        if len(ref_imgs) == 0:
            print(f"Aborting -- Latent {latent_idx} is never activated.")
            return

        ref_masked_imgs = self.get_masked_image(ref_imgs, latent_idx)
        ref_masked_imgs = np.clip(ref_masked_imgs, 0, 1)
        ref_masked_imgs = [Image.fromarray((img.numpy() * 255).astype(np.uint8)) for img in ref_masked_imgs]  # Convert tensors to PIL images

        if input_img is not None:
            input_img = input_img.resize((resize_size, resize_size))
            input_masked = self.get_masked_image([input_img], latent_idx).squeeze(0)
            input_masked = np.clip(input_masked, 0, 1)
            input_masked = Image.fromarray((input_masked.numpy() * 255).astype(np.uint8))  # Convert tensor to PIL image

        num_images = len(ref_imgs) + (1 if input_img is not None else 0)

        combined_image_height = resize_size * 2  # Added height for title
        combined_image = Image.new("RGB", (resize_size * num_images, 50 + combined_image_height))

        # Draw the title on the combined image
        draw = ImageDraw.Draw(combined_image)
        draw.text((10, 10), description, fill="white", font_size=30)  # Draw the description at the top left

        x_offset = 0
        if input_img is not None:
            combined_image.paste(input_img, (0, 50))  # Adjusted y-offset for input_masked
            x_offset = input_img.width

        for img in ref_imgs:
            combined_image.paste(img, (x_offset, 50))  # Adjusted y-offset for reference images
            x_offset += img.width

        x_offset = 0
        if input_img is not None:
            combined_image.paste(input_masked, (x_offset, 50 + resize_size))  # Adjusted y-offset for masked reference images
            x_offset += input_masked.width

        for masked_img in ref_masked_imgs:
            combined_image.paste(masked_img, (x_offset, 50 + resize_size))  # Adjusted y-offset for masked reference images
            x_offset += masked_img.width

        return combined_image
