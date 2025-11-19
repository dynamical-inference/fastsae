from datetime import datetime
import os
import pathlib

import datasets
from dotenv import load_dotenv
import fastsae
import shortuuid
import torch
from transformers import AutoProcessor

if __name__ == "__main__":
    load_dotenv()

    ######################
    # config
    ######################
    dataset_name = "evanarlian/imagenet_1k_resized_256"
    dataset_split = "train"
    collate = "auto"
    device = "cuda"

    # backbone architecture
    backbone_name = "openai/clip-vit-base-patch16"
    backbone_layer = "vision_model.encoder.layers.10"

    # sae architecture
    d_sae_factor = 64
    architecture = "vanilla"
    activation = "relu"

    # sae training
    sparsity_coefficient = 0.00008
    learning_rate = 0.0004
    scheduler_name = "constantwithwarmup"
    warm_up_steps = 500
    batch_size = 256  # image batch size
    total_n_training_samples = 2_621_440
    total_n_checkpoints = 10  # default 10
    use_ghost_grads = True
    sae_b_dec_init_method = "geometric_median"
    stats_top_k_max_samples = 100

    # dataloader
    _caching = False

    # verbose
    _verbose = True

    # wandb
    _wandb_enabled = os.getenv("WANDB_ENABLED", "false").lower() == "true"
    _wandb_project = os.getenv("WANDB_PROJECT", None)
    _wandb_entity = os.getenv("WANDB_ENTITY", None)

    ######################
    # model
    ######################

    # create backbone
    backbone = fastsae.models.Backbone(model_hf_name=backbone_name, hook_target_name=backbone_layer, device=device)
    d_in = backbone.hook_target_dim

    # create sae
    sae_instance = fastsae.models.SAE(
        d_in=d_in,
        d_sae_factor=d_sae_factor,
        architecture=architecture,
        activation_name=activation,
        device=device,
    )

    # create wrapper
    wrapper = fastsae.models.SAEWrapper(
        backbone=backbone,
        sae=sae_instance,
        _caching=_caching,
    )

    ######################
    # dataloader
    ######################

    # dataloader, auto_processor
    auto_processor = AutoProcessor.from_pretrained(backbone_name)

    def processor(img):
        return auto_processor(images=img, text="a photo", return_tensors="pt")

    dataset_instance = datasets.load_dataset(
        path=dataset_name,
        split=dataset_split,
    )
    dataloader = fastsae.dataset.flexible.create_dataloader(
        dataset=dataset_instance,
        processor=processor,
        return_pixel_values_only=True,
        return_labels_np=False,
        batch_size=batch_size,
        shuffle=True,
        collate=collate,
    )

    ######################
    # train
    ######################

    # set ckpt_path
    _run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{shortuuid.uuid()[:6]}"
    ckpt_path = f"{os.getenv('CHECKPOINT_PATH', './checkpoints')}/{_run_id}"

    # train sae
    optimizer = torch.optim.Adam(wrapper.sae.parameters(), lr=learning_rate)
    scheduler = fastsae.training.util.get_scheduler(scheduler_name, optimizer=optimizer, warm_up_steps=warm_up_steps)

    wrapper.backbone.train()
    wrapper.sae.train()

    wrapper.train(
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        total_n_training_samples=total_n_training_samples,
        sparsity_coefficient=sparsity_coefficient,
        use_ghost_grads=use_ghost_grads,
        sae_b_dec_init_method=sae_b_dec_init_method,
        checkpoint_path=ckpt_path,
        total_n_checkpoints=total_n_checkpoints,
        log_to_wandb=_wandb_enabled,
        wandb_run_id=_run_id,
        wandb_project=_wandb_project,
        wandb_entity=_wandb_entity,
    )

    ######################
    # compute statistics
    ######################

    if _verbose:
        print("Computing statistics...")

    stats_path = pathlib.Path(ckpt_path, "final", "stats.split_train.seed_no")
    if not stats_path.exists():
        stats_path.mkdir(parents=True, exist_ok=False)
    else:
        if _verbose:
            print(f"Statistics already computed in {stats_path}. Terminating.")
        exit()

    dataloader_stats, labels_np = fastsae.dataset.flexible.create_dataloader(
        dataset=dataset_instance,
        processor=auto_processor,
        return_pixel_values_only=False,
        return_labels_np=True,
        batch_size=batch_size,
        shuffle=False,
        collate=collate,
    )

    intrp = fastsae.interpret.SAEInterpreter(
        wrapped=wrapper,
        backbone_processor=auto_processor,
        stats_dataloader=dataloader_stats,
        stats_labels_np=labels_np,
        stats_top_k_max_samples=stats_top_k_max_samples,
    )

    intrp.compute_stats(compute_cls_wise_top_idx=True)
    intrp.save_stats(stats_path)
