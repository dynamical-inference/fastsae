import argparse
from datetime import datetime
import json
import os
import pathlib
import sys

import shortuuid

sys.path.append("../../")

import datasets
from dotenv import load_dotenv
import fastsae
from fastsae.downstream_tasks.openai_imagenet_templates import openai_imagenet_template
from fastsae.utils.hub import load_from_hub
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import transformers


def get_text_features_openai(classnames, processor, sae_wrapper):
    """Calculate mean text features across templates for each class."""
    mean_text_features = 0

    for template_fn in openai_imagenet_template:
        # Generate prompts and convert to token IDs
        prompts = [template_fn(c) for c in classnames]
        prompt_ids = [processor(text=p, return_tensors="pt", padding=False, truncation=True).input_ids[0] for p in prompts]

        # Process batch
        padded_prompts = pad_sequence(prompt_ids, batch_first=True).to(sae_wrapper.device)

        # Get text features
        with torch.no_grad():
            text_features = sae_wrapper.backbone.model.get_text_features(padded_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features += text_features

    return mean_text_features / len(openai_imagenet_template)


def get_predictions(out, text_features, sae_wrapper):
    image_features = out.image_embeds
    logit_scale = sae_wrapper.backbone.model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.t()
    preds = logits.argmax(dim=-1)
    return preds


def get_blocked_latent_indices(intrp):
    assert "cls_wise_top_idx" in intrp.stats
    sorted_items = sorted(intrp.stats["cls_wise_top_idx"].items())
    cls_wise_top_idx_np = np.stack([v.cpu().numpy() for k, v in sorted_items])
    value_counts = np.bincount(cls_wise_top_idx_np.flatten())
    sorted_value_counts = np.sort(value_counts)[::-1]
    sorted_indices = np.argsort(value_counts)[::-1]
    blocked_latent_indices = sorted_indices[sorted_value_counts > 500]
    return torch.tensor(blocked_latent_indices, dtype=torch.long)


def excluded_topk_per_label(labels: torch.Tensor, stats: dict, blocked: torch.Tensor, k: int) -> torch.Tensor:
    # labels: LongTensor [B]
    # stats['cls_wise_top_idx'][label] -> LongTensor [K_stats] (sorted by importance)
    per_label_lists = []
    for y in labels.tolist():
        cls_top = stats["cls_wise_top_idx"][int(y)]
        # filter out any blocked indices, keep ordering
        mask = ~torch.isin(cls_top, blocked.to(cls_top.device))
        filtered = cls_top[mask]
        per_label_lists.append(filtered)

    # ensure consistent width across the batch
    k_eff = min(k, min(t.numel() for t in per_label_lists))
    if k_eff == 0:
        raise ValueError("No available indices after exclusion. Reduce blocked indices or k.")

    return torch.stack([t[:k_eff] for t in per_label_lists], dim=0).to(dtype=torch.long)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Classification ablation on ImageNet-1K")
    parser.add_argument(
        "--local_ckpt_path",
        type=str,
        help="Path to the trained model checkpoint directory (containing 'backbone' and 'final')",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to compute statistics on",
    )
    parser.add_argument(
        "--ablate_topk",
        type=int,
        default=10,
        help="Number of top k latents to ablate",
    )
    args = parser.parse_args()

    print(f"Available GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")

    ######################
    # config
    ######################

    dataset_name = "evanarlian/imagenet_1k_resized_256"
    classnames_path = "../../fastsae-lib/downstream_tasks/imagenet_classnames.txt"
    dataset_split = args.dataset_split
    collate = "auto"
    device = "cuda"

    # dataloader
    _caching = True if dataset_split == "val" else False

    # verbose
    _verbose = True

    if args.local_ckpt_path is not None:
        print("Loading model from local path")
        ckpt_path = args.local_ckpt_path
        backbone = fastsae.models.Backbone.load(os.path.join(ckpt_path, "backbone"))
        sae = fastsae.models.SAE.load(os.path.join(ckpt_path, "final"))
        wrapper = fastsae.models.SAEWrapper(backbone=backbone, sae=sae)

    else:
        print("Loading model from Hugging Face Hub")
        wrapper, ckpt_path = load_from_hub(
            repo_id="hyesulim/fastsae-models",
            cache_dir="./hf-fastsae-models",  # If you want to specify a different cache directory
            return_ckpt_path=True,
        )

    wrapper.backbone.eval()
    wrapper.sae.eval()

    dataset_instance = datasets.load_dataset(
        path=dataset_name,
        split=dataset_split,
    )
    auto_processor = transformers.AutoProcessor.from_pretrained(wrapper.backbone.config["model_hf_name"])

    dataloader_eval = fastsae.dataset.flexible.create_dataloader(
        dataset=dataset_instance,
        processor=auto_processor,
        return_pixel_values_only=False,  # NOTE: Keep this False for classification
        return_label=True,  # NOTE: Keep this True for classification to compute accuracy
        batch_size=256,
        shuffle=False,
        collate="auto",
    )

    intrp = fastsae.interpret.SAEInterpreter(
        wrapped=wrapper,
        backbone_processor=auto_processor,
        stats_dataloader=dataloader_eval,
    )
    intrp.load_stats(pathlib.Path(ckpt_path, "final", f"stats.split_{dataset_split}.seed_no"))

    with open(classnames_path, "r") as file:
        classnames = [" ".join(line.strip().split(" ")[1:]) for line in file.readlines()]

    text_features = get_text_features_openai(classnames, auto_processor, wrapper)
    print("Text features calculated using OpenAI templates")

    blocked_latent_indices = get_blocked_latent_indices(intrp)
    k = args.ablate_topk

    print(f"Steering class-wise top {k} latents set to 0")

    acc_org = 0
    acc_steer = 0
    n_samples = 0

    for x in tqdm(dataloader_eval):
        if k > 0:
            indices = excluded_topk_per_label(x["label"], intrp.stats, blocked_latent_indices, k)
            steer_cfg = {"indices": indices, "set": 0}
        else:
            steer_cfg = {"indices": range(wrapper.sae.d_sae), "multiply": 1}
        out_pre, out_post, a = wrapper(x, steer_cfg=steer_cfg)
        preds_pre = get_predictions(out_pre, text_features, wrapper).cpu()
        preds_post = get_predictions(out_post, text_features, wrapper).cpu()
        acc_org += (preds_pre == x["label"]).float().sum()
        acc_steer += (preds_post == x["label"]).float().sum()
        n_samples += len(x["label"])

        print(f"[{n_samples}/{len(dataloader_eval.dataset)}] avg acc_org: {acc_org / n_samples}, avg acc_steer: {acc_steer / n_samples}")

    print(f"Final avg acc_org: {acc_org / n_samples}")
    print(f"Final avg acc_steer: {acc_steer / n_samples}")

    # save results as a csv file
    results = {
        "acc_org_avg": (acc_org / n_samples).item(),
        "acc_steer_avg": (acc_steer / n_samples).item(),
        "ablate_topk": k,
        # "acc_per_class_org": acc_per_class_org,
        # "acc_per_class_steer": acc_per_class_steer,
    }

    results_dir = pathlib.Path(ckpt_path, "final", "classification_ablation")
    os.makedirs(results_dir, exist_ok=True)
    file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_split}_{shortuuid.uuid()}.json"
    json.dump(results, open(results_dir / file_name, "w"))
    print(f"Results saved to {results_dir / file_name}")
