# Can World Models Benefit VLMs for World Dynamics?

<p align="left">
  <a href="https://arxiv.org/abs/2506.18897">
    <img src="https://img.shields.io/badge/arXiv-2510.00855-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://dyva-worldlm.github.io">
    <img src="https://img.shields.io/badge/Project Page-DyVA--WorldLM-blue" alt="Project Website">
  </a>
  <a href="https://huggingface.co/dyva/DyVA-WorldLM">
    <img src="https://img.shields.io/badge/Model Checkpoint-DyVA--WorldLM-6aa84f" alt="Model Checkpoint">
  </a>
</p>

<p align="left">
    <a href="https://github.com/zyzkevin">Kevin Zhang<sup>1,*</sup></a> • <a href="https://kuangzhige.github.io/">Kuangzhi Ge<sup>1,*</sup></a> • <a href="https://litwellchi.github.io/">Xiaowei Chi<sup>2,*</sup></a> • <a href="https://zrrskywalker.github.io/">Renrui Zhang<sup>3</sup></a> • <a href="">Shaojun Shi<sup>1</sup></a> • <a href="https://dong-zhen.com/">Zhen Dong<sup>4</sup></a> • <a href="https://siruihan.com/">Sirui Han<sup>2,†</sup></a> • <a href="https://www.shanghangzhang.com/">Shanghang Zhang<sup>1,†</sup></a>
</p>

<sup>1</sup> Peking University  <sup>2</sup> Hong Kong University of Science and Technology  <sup>3</sup> Chinese University of Hong Kong  <sup>4</sup> University of California, Santa Barbara

\* Equal Contribution | † Corresponding Author

---

## Overview

[**Installation**](#installation) | [**Inference**](#inference) | [**Training**](#training) | [**Code Structure**](#code-structure)

Modern VLMs handle static images well but struggle with dynamics. Generative world models, however, capture rich temporal structure through video prediction.

We unify the two by introducing **WorldLMs**, which use a video diffusion model as a generative encoder. A single denoising step injects dynamics-aware latents into the VLM, enabling reasoning beyond static input.

Our best model, **DyVA**, greatly improves spatial–temporal reasoning and even lets single-image VLMs perform multi-frame reasoning. Our method features：

- Single-step world model query → dynamics-aware latent representation
- Single image → multi-frame reasoning via dynamic latent alignment
- State-of-the-art or competitive performance on challenging dynamic reasoning tasks
- Modular, plug-and-play design compatible with existing VLMs

---

## Installation

We recommend using Conda to manage the environment. In our experiments and training, we use `python3.11` and `flash-attn2.8.0`. You can install the required environment for inference and training as follow:

```bash
git clone https://github.com/zyzkevin/dyva-worldlm.git
cd dyva-worldlm

conda env create -f environment.yml
conda activate dyva-worldlm

# Training additionally requires Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install flash-attn --no-build-isolation
```

---

## Usage

### Inference

Once installed, loading and running inference with our pretrained models as follows:

```python
import torch
from PIL import Image
from prismatic import load
import os

# For gated LMs like Llama-2, ensure you have a Hugging Face token
hf_token = "YOUR_HF_KEY"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (e.g., DyVA-7B)
model_id = "dyva_siglip_qwen+7b" # "dyva_siglip+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

# Open an image
image_path = "./assets/rocket.png" # Replace with your image
image = Image.open(image_path).convert("RGB")

# Build prompt
user_prompt = "What is in the image and what is going on?"
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate
generated_text = vlm.generate(
image,
prompt_text,
do_sample=True,
temperature=0.4,
max_new_tokens=1024,
min_length=1,
)

print(f"Generated Text: {generated_text}\n")
```

### Training

Training is orchestrated by `scripts/pretrain.py` and is best launched using `torchrun`.

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs, as is implemented in prismatic-vlms.  Here is an example to train `DyVA-siglip+7B` with 8 GPUs:

```bash
torchrun --nnodes 1 --nproc-per-node 8 \
scripts/pretrain.py
--model.type "dyva_siglip" \
--model.model_id "dyva_siglip" \
--model.arch_specifier "no-align+dual" \
--model.llm_backbone_id "llama2-7b-pure" \
--model.llm_max_length 2048 \
--model.image_resize_strategy "resize-naive" \
--model.enable_mixed_precision_training True \
--model.finetune_per_device_batch_size 1 \
--num_frames 8 \
--model.finetune_learning_rate 1e-5 \
--hf_token HF_TOKEN \
--run_id "example" \
--run_root_dir "runs" \
--dataset.type "llava_svd" \
--wandb_project YOUR_WANDB_PROJECT \
--wandb_entity YOUR_ENTITY \
--stage "finetune"
```

---

## Code Structure
High-level overview of dyva-worldlm file-tree:

- dyva-worldlm/
    - prismatic/ - Core library code (based on prismatic-vlms)
    - scripts/ - High-level runnable scripts
    - assets/ - Stores materials used for demos and examples.
    - train.sh - Example training launch script
    - infer.py - Minimal inference example
    - environment.yml - Conda environment file
    - README.md


---


## Acknowledgements
This repository is based on the excellent [**Prismatic VLMs**](https://github.com/TRI-ML/prismatic-vlms) codebase. We extend our sincere gratitude to the original authors for their foundational work and well-structured code.

---

## Citation
If you find our work useful in your research, please consider citing our paper:

```bibtex
@misc{zhang2025worldmodelsbenefitvlms,
      title={Can World Models Benefit VLMs for World Dynamics?},
      author={Kevin Zhang and Kuangzhi Ge and Xiaowei Chi and Renrui Zhang and Shaojun Shi and Zhen Dong and Sirui Han and Shanghang Zhang},
      year={2025},
      eprint={2510.00855},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.00855},
}
```
