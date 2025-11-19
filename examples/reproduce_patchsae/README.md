## Reproducing PatchSAE with the FastSAE package
> PatchSAE (ICLR 2025) repo: https://github.com/dynamical-inference/patchsae

1. Getting started
    - Follow "Getting Started" in the FastSAE package README.

2. Follow [`tutorial.ipyb`](tutorial.ipynb). This shows how to use FastSAE package by reproducing [PatchSAE](https://github.com/dynamical-inference/patchsae) paper.

   
## Load from Huggingface
> https://huggingface.co/hyesulim/fastsae-models
```python
from fastsae.utils.hub import load_from_hub
wrapper = load_from_hub(repo_id="hyesulim/fastsae-models")
```

## Advanced
### Training SAE
```bash
cd sae
PYTHONPATH=. python examples/reproduce_patchsae/train_patchsae.py
```

- Note: With the default config (batch size 256), expect approximately up to 42 GB GPU memory and ~6 hours on a single A6000.
- Reference training log: [wandb log](https://api.wandb.ai/links/hyesulim-hs/wk4shoc6)
- Outcome: backbone config, SAE checkpoint, config, and stats

### Computing statistics

If you trained your SAE using `python examples/reproduce_patchsae/train_patchsae.py`, then stats are already computed for the final checkpoint.


Run this if you want to compute stats for different checkpoints.
```bash
cd sae
PYTHONPATH=. python examples/reproduce_patchsae/compute_stats.py --local_ckpt_path /absolute/path/to/your/checkpoint --dataset_split val
```

### Downstream task: ImageNet-1K classification

Run ImageNet-1K classification with class-wise topk masking ablation.
```bash
cd sae/examples/reproduce_patchsae
PYTHONPATH=. python classification_ablation.py --local_ckpt /absolute/path/to/your/checkpoint --dataset_split val --ablate_topk 10
```
You will get something like:

When masking out class-wise top 10 latents:
```json
{"acc_org_avg": 0.6729199886322021, "acc_steer_avg": 0.29175999760627747, "ablate_topk": 10}
```

When replacing with SAE reconstruction without masking:
```json
{"acc_org_avg": 0.6729199886322021, "acc_steer_avg": 0.6334400177001953, "ablate_topk": 0}
```
