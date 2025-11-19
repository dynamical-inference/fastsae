import argparse
import os
import pathlib

import datasets
from dotenv import load_dotenv
import fastsae
from transformers import AutoProcessor

if __name__ == "__main__":
    load_dotenv()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    parser = argparse.ArgumentParser(description="Compute SAE statistics")
    parser.add_argument(
        "--local_ckpt_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint directory (containing 'backbone' and 'final')",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to compute statistics on",
    )
    args = parser.parse_args()

    ######################
    # config
    ######################
    dataset_name = "evanarlian/imagenet_1k_resized_256"
    dataset_split = args.dataset_split
    collate = "auto"
    device = "cuda"

    # dataloader
    _caching = True if args.dataset_split == "val" else False

    # verbose
    _verbose = True

    ######################
    # model
    ######################

    ## Option 1: Download from HuggingFace Hub
    # wrapper, ckpt_path = load_from_hub(
    #     repo_id="hyesulim/fastsae-models",
    #     cache_dir="./hf-fastsae-models", # If you want to specify a different cache directory
    #     return_ckpt_path=True,
    # )

    ## Option 2: Use local path
    ckpt_path = args.local_ckpt_path
    backbone = fastsae.models.Backbone.load(os.path.join(ckpt_path, "backbone"))
    sae = fastsae.models.SAE.load(os.path.join(ckpt_path, "final"))
    wrapper = fastsae.models.SAEWrapper(backbone=backbone, sae=sae)

    backbone_name = wrapper.backbone.config["model_hf_name"]

    ######################
    # dataloader
    ######################

    # dataloader, auto_processor
    auto_processor = AutoProcessor.from_pretrained(backbone_name)
    dataset_instance = datasets.load_dataset(
        path=dataset_name,
        split=dataset_split,
    )

    dataloader, labels_np = fastsae.dataset.flexible.create_dataloader(
        dataset=dataset_instance,
        processor=auto_processor,
        return_pixel_values_only=False,  # NOTE: Keep this False for classification
        return_label=True,  # NOTE: Keep this True for classification to compute accuracy
        return_labels_np=True,
        batch_size=512,
        shuffle=False,
        collate="auto",
    )

    stats_path = pathlib.Path(ckpt_path, "final", f"stats.split_{dataset_split}.seed_no")
    if not stats_path.exists():
        stats_path.mkdir(parents=True, exist_ok=False)
    else:
        if _verbose:
            print(f"Statistics already computed in {stats_path}. Terminating.")
        exit()

    intrp = fastsae.interpret.SAEInterpreter(
        wrapped=wrapper,
        backbone_processor=auto_processor,
        stats_dataloader=dataloader,
        stats_labels_np=labels_np,
    )
    try:
        intrp.compute_stats(compute_cls_wise_top_idx=True)

    finally:
        intrp.save_stats(stats_path)
