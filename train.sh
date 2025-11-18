# !/bin/bash
# Set environment variables
export WANDB_API_KEY="YOUR_WANDB_KEY"
export HF_TOKEN="YOUR_HF_KEY"
export HF_HOME="YOUR_HF_HOME"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TRAINING CONFIGURATIONS
nodes_num=1
nproc_per_node=2
batch_size=8
num_frames=8
finetune_epochs=1
stage="finetune"

# MODEL CONFIGURATIONS
vision_backbone_type="dyva_siglip"
llm_backbone_id="qwen25-7b-chat" # "llama2-7b-pure"
image_resize_strategy="resize-naive"
llm_max_length=2048
arch_specifier="no-align+dual" # "no-align+dual" # "align+fused-gelu-mlp"
dataset_type="llava-svd"

# WandB CONFIGURATIONS
wandb_entity="YOUR_ENTITY"
wandb_project="dyva-worldlm"

# Determine model_id and run_root based on backbone types
if [[ "$llm_backbone_id" == "llama2-7b-pure" ]]; then
    llm=""
elif [[ "$llm_backbone_id" == qwen25-7b-* ]]; then
    llm="_qwen"
else
    echo "Unsupported llm_backbone_id: $llm_backbone_id"
    exit 1
fi

model_id="${vision_backbone_type}${llm}+7b"
finetune_run_id="${model_id}_ft_proj_noalign"
echo $model_id

torchrun --nnodes $nodes_num --nproc-per-node $nproc_per_node \
    scripts/pretrain.py \
    --model.type "$model_id" \
    --model.model_id "$model_id" \
    --model.arch_specifier $arch_specifier \
    --model.llm_backbone_id $llm_backbone_id \
    --model.llm_max_length $llm_max_length \
    --model.image_resize_strategy $image_resize_strategy \
    --model.enable_mixed_precision_training True \
    --model.finetune_per_device_batch_size 1 \
    --num_frames $num_frames \
    --model.finetune_learning_rate 1e-5 \
    --hf_token HF_TOKEN \
    --run_id "$finetune_run_id" \
    --run_root_dir "runs" \
    --dataset.type "$dataset_type" \
    --wandb_project "$wandb_project" \
    --wandb_entity "$wandb_entity" \
    --stage "$stage" \