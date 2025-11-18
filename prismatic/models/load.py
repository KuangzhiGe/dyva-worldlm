"""
load.py

Entry point for loading pretrained VLMs for inference.
Modified to support custom models from dyva/DyVA-WorldLM.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download

from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)

# === HF Hub Repositories ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
DYVA_REPO = "dyva/DyVA-WorldLM"

# === Custom Models ===
DYVA_MODELS = {
    "dyva_siglip_qwen+7b",
    "dyva_siglip+7b"
}

# === Available Models ===
def available_model_ids() -> List[str]:
    return list(MODEL_REGISTRY.keys()) + list(DYVA_MODELS)

def available_model_ids_and_names() -> List[List[str]]:
    return list(GLOBAL_REGISTRY.values())

def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        # Custom models might not have descriptions in the registry
        if model_id_or_name in DYVA_MODELS:
            return f"Custom DyVA model: {model_id_or_name}"
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))
    return description

# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path], hf_token: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from local disk, default HF Hub, or DyVA HF Hub."""
    print(f"Loading: {model_id_or_path}")

    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")
        config_json = run_dir / "config.json"
        checkpoint_pt = run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        # Determine Repo and Model ID
        if model_id_or_path in DYVA_MODELS:
            repo_id = DYVA_REPO
            model_id = model_id_or_path # Use the name directly as ID/Folder name
            overwatch.info(f"Downloading `{model_id}` from Custom HF Hub: {repo_id}")
        elif model_id_or_path in GLOBAL_REGISTRY:
            repo_id = HF_HUB_REPO
            model_id = GLOBAL_REGISTRY[model_id_or_path]['model_id']
            overwatch.info(f"Downloading `{model_id}` from Default HF Hub: {repo_id}")
        else:
            raise ValueError(f"Couldn't find `{model_id_or_path}`; check available models.")

        # Download files
        config_json = hf_hub_download(repo_id=repo_id, filename=f"{model_id}/config.json", cache_dir=cache_dir, token=hf_token)
        checkpoint_pt = hf_hub_download(
            repo_id=repo_id, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir, token=hf_token
        )

    # Load Model Config
    with open(config_json, "r") as f:
        cfg_whole = json.load(f)
        model_cfg = cfg_whole["model"]

    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        cfg_whole.get("num_frames", 8),
    )

    # Load LLM Backbone
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token, # Use the argument passed to load(), do not hardcode secrets.
        inference_mode=False, # Maintained per user request
    )

    # Load VLM
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint; Freezing Weights 🥶")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
    )

    return vlm