# fastsae/hub.py
import os
from pathlib import Path

import fastsae
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "hyesulim/fastsae-weights"  # replace with your repo id
DEFAULT_REVISION = None  # or "v1", commit hash, etc.


def _snapshot(repo_id: str = DEFAULT_REPO_ID, revision: str | None = DEFAULT_REVISION, cache_dir: str | None = None, token: str | None = None) -> Path:
    """
    Download the HF repo snapshot (cached) and return local path.
    If user already provided a local path in FASTSAE_LOCAL_WEIGHTS, use that instead.
    """
    # Allow users to override with local dir env var (helpful for tests/offline)
    local_override = os.environ.get("FASTSAE_LOCAL_WEIGHTS")
    if local_override:
        p = Path(local_override)
        if p.exists():
            return p
        # else fall through to download

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=False,  # change to True if you want strict offline behavior
    )
    return Path(local_dir)


def load_from_hub(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = DEFAULT_REVISION,
    token: str | None = None,
    cache_dir: str | None = None,
    return_ckpt_path: bool = False,
):
    """
    High-level helper: downloads snapshot (if needed), loads backbone + sae and returns wrapper.
    """
    repo_path = _snapshot(repo_id=repo_id, revision=revision, token=token, cache_dir=cache_dir)
    backbone_path = repo_path / "backbone"
    sae_path = repo_path / "final"

    backbone = fastsae.models.Backbone.load(backbone_path)
    sae_inst = fastsae.models.SAE.load(sae_path)
    if return_ckpt_path:
        return fastsae.models.SAEWrapper(backbone=backbone, sae=sae_inst), repo_path
    return fastsae.models.SAEWrapper(backbone=backbone, sae=sae_inst)
