export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN="YOUR_HF_KEY"
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vision_backbone_type="svd_dual_siglip"
batch_size=8
nnodes_num=8
num_frames=8
total_processes=16
align_epochs=1
finetune_epochs=1
model_id="prism-${vision_backbone_type}"
finetune_run_id="${model_id}_ft_proj_noalign_qwen"
image_resize_strategy="resize-naive"
llm_max_length=2048
llm_backbone_id="qwen25-7b-pure" # "llama2-7b-pure" "qwen25-7b-pure"
arch_specifier="no-align+dual" # "align+dual" # "align+fused-gelu-mlp"
wandb_entity="2200013209-peking-university"
dataset_type="llava-svd_multi"

if [[ "${vision_backbone_type}" == "svd" || "${vision_backbone_type}" == "svd_unet" ]]; then
    run_root="svd_only"
elif [[ "${vision_backbone_type}" == svd_* || "${vision_backbone_type}" == svd_dual_* ]]; then
    run_root="${vision_backbone_type/svd_dual_/svd_}"
else
    run_root="${vision_backbone_type}"
    echo "Warning: vision_backbone_type不符合预设规则，run_root已设置为'${run_root}'"
fi
echo $run_root

torchrun --nnodes 1 --nproc-per-node 8 \
    scripts/pretrain.py \
    --model.type "$model_id" \
    --model.model_id "$model_id" \
    --model.arch_specifier $arch_specifier \
    --model.llm_max_length $llm_max_length \
    --model.image_resize_strategy $image_resize_strategy \
    --model.enable_mixed_precision_training True \
    --model.finetune_per_device_batch_size 1 \
    --num_frames $num_frames \
    --model.finetune_learning_rate 1e-5 \
    --hf_token HF_TOKEN \
    --run_id "$finetune_run_id" \
    --run_root_dir "runs/${run_root}" \
    --dataset.type "$dataset_type" \
    --wandb_project "svd_vlm" \
    --wandb_entity "$wandb_entity" \
    --stage "finetune" \