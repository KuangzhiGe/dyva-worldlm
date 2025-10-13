export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000 
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN="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"
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TORCH_USE_CUDA_DSA=1 torchrun --nnodes 1 --nproc-per-node 2 scripts/pretrain.py \
  --model.type "prism-svd" \
  --model.model_id "prism-svd" \
  --model.vision_backbone_id "clip-svd" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "llama2-7b-pure" \
  --model.arch_specifier "align+gelu-mlp" \
  --model.enable_mixed_precision_training True \
  --model.finetune_per_device_batch_size 2 \
  --model.finetune_epochs 100000 \
  --dataset.type "llava-svd" \
  --hf_token HF_TOKEN \
  --run_id "finetune_fake" \
  --wandb_project "svd_vlm" \
  --wandb_entity "2200013209-peking-university" \
  --stage "finetune" \
  --pretrained_checkpoint "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/fake/align/checkpoints/latest-checkpoint.pt" \