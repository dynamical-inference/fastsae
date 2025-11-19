# 🏎️ FastSAE: Adopt seamlessly, train fast

## 🛠 Getting Started

Follow these steps to install and verify FastSAE.

```bash
# 1) Create and activate a fresh environment
conda create -n fastsae python=3.12 -y
conda activate fastsae

# 2) Install project dependencies
pip install -r requirements.txt

# 3) Install PyTorch matching your system (CUDA/CPU)
#    See official instructions at `https://pytorch.org/get-started/locally/`
#    (If you already have a working torch install, you can skip this.)

# 4) Install FastSAE in editable mode
pip install -e .

# 5) (Optional) Dev tools
pip install -U pre-commit ruff
pre-commit install
```

Create a `.env` file in the repo root to configure paths and runtime behavior. <br>
Or use the provided example to create your `.env`:

```bash
cp env.example .env
```

Quick checks:

```bash
# Verify install and version
python -c "import fastsae, torch; print('fastsae', fastsae.__version__, '| cuda:', torch.cuda.is_available())"
```

## 🙌 Onboarding 
👉 Follow [`examples/reproduce_patchsae/tutorial.ipynb`](examples/reproduce_patchsae/tutorial.ipynb). <br>
This shows how to use FastSAE package by reproducing [PatchSAE](https://github.com/dynamical-inference/patchsae) paper.

## 🙏 Citation
If you find our code or models useful in your work, please cite our paper:
```
@inproceedings{
  lim2025patchsae,
  title={Sparse autoencoders reveal selective remapping of visual concepts during adaptation},
  author={Hyesu Lim and Jinho Choi and Jaegul Choo and Steffen Schneider},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=imT03YXlG2}
}
```
