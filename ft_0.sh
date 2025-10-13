export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN="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"
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vision_backbone_type="svd_dual_siglip"
master_addr=10.40.0.130
master_port=12134
rank=0
batch_size=8
nnodes_num=8
num_frames=8
total_processes=16
align_epochs=1
finetune_epochs=1
model_id="prism-${vision_backbone_type}"
align_run_id="${model_id}_align_proj"
finetune_run_id="${model_id}_ft_proj_noalign_qwen"
multi_run_id="${model_id}_ft_proj_multi_noalign_qwen"
image_resize_strategy="resize-naive" # "anyres_max_9" "resize-naive"
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

# Align
# torchrun --nnodes 2 --nproc-per-node 8 \
#   --node_rank=$rank \
#   --master_addr=$master_addr \
#   --master_port=$master_port \
#   scripts/pretrain.py \
#   --model.type "$model_id" \
#   --model.model_id "$model_id" \
#   --model.arch_specifier $arch_specifier \
#   --model.llm_max_length $llm_max_length \
#   --model.image_resize_strategy $image_resize_strategy \
#   --model.enable_mixed_precision_training True \
#   --model.align_per_device_batch_size $batch_size \
#   --num_frames $num_frames \
#   --hf_token HF_TOKEN \
#   --run_id "$align_run_id" \
#   --run_root_dir "runs/${run_root}" \
#   --dataset.type "llava-svd" \
#   --wandb_project "svd_vlm" \
#   --wandb_entity "$wandb_entity" \
#   --stage "align"


# Finetune
# torchrun --nnodes 2 --nproc-per-node 8 \
#   --node_rank=$rank \
#   --master_addr=$master_addr \
#   --master_port=$master_port \
#   scripts/pretrain.py \
#   --model.type "$model_id" \
#   --model.model_id "$model_id" \
#   --model.arch_specifier $arch_specifier \
#   --model.llm_max_length $llm_max_length \
#   --model.image_resize_strategy $image_resize_strategy \
#   --model.enable_mixed_precision_training True \
#   --model.finetune_per_device_batch_size $batch_size \
#   --num_frames $num_frames \
#   --hf_token HF_TOKEN \
#   --run_id "$finetune_run_id" \
#   --run_root_dir "runs/${run_root}" \
#   --dataset.type "llava-svd" \
#   --wandb_project "svd_vlm" \
#   --wandb_entity "$wandb_entity" \
#   --stage "finetune" \
#   --pretrained_checkpoint "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/${run_root}/${align_run_id}/checkpoints/latest-checkpoint.pt"

# multi-image: grad_acc_step=2=global//per_device//num_gpus, global_batchsize=128
torchrun --nnodes 2 --nproc-per-node 8 \
  --node_rank=$rank \
  --master_addr=$master_addr \
  --master_port=$master_port \
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
  --run_id "$multi_run_id" \
  --run_root_dir "runs/${run_root}" \
  --dataset.type "$dataset_type" \
  --wandb_project "svd_vlm" \
  --wandb_entity "$wandb_entity" \
  --stage "finetune" \
  --pretrained_checkpoint "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/${run_root}/${finetune_run_id}/checkpoints/latest-checkpoint.pt"

# No-Align
# arch_specifier="no-align+fused-gelu-mlp" # "no-align+dual" no-align+fused-gelu-mlp
# torchrun --nnodes 2 --nproc-per-node 8 \
#   --node_rank=$rank \
#   --master_addr=$master_addr \
#   --master_port=$master_port \
#   scripts/pretrain.py \
#   --model.type "$model_id" \
#   --model.model_id "$model_id" \
#   --model.arch_specifier $arch_specifier \
#   --model.enable_mixed_precision_training True \
#   --model.finetune_per_device_batch_size $batch_size \
#   --hf_token HF_TOKEN \
#   --run_id "$finetune_run_id" \
#   --run_root_dir "runs/${run_root}" \
#   --dataset.type "$dataset_type" \
#   --stage "finetune-svd" \
#   --wandb_project "svd_vlm" \
#   --wandb_entity "$wandb_entity" \

  # --pretrained_checkpoint "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/${run_root}/${align_run_id}/checkpoints/latest-checkpoint.pt" \
  # --align_svd_with_llm False \
  # --align_weight 0.1


cd /mnt/world_foundational_model/gkz/prismatic-vlms/vlm-evaluation
model_dir="/mnt/world_foundational_model/gkz/prismatic-vlms/runs/${run_root}/${finetune_run_id}"
data_dir="/mnt/world_foundational_model/gkz/data/vlm-evaluation"

accelerate launch \
  --num_machines=2 \
  --num_processes=$total_processes \
  --main_process_ip=$master_addr \
  --main_process_port=$master_port \
  --machine_rank=$rank \
  scripts/evaluate.py \
  --model_family prismatic \
  --model_id "${model_id}" \
  --dataset_root_dir "$data_dir" \
  --results_dir "${model_dir}/results" \
  --model_dir "$model_dir" \
  --dataset_types ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]

python scripts/score.py --model_id "${model_id}" --dataset_root_dir "$data_dir" --results_dir "${model_dir}/results" --dataset_types ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]


# Fake Train
cd /mnt/world_foundational_model/gkz/prismatic-vlms

TORCH_USE_CUDA_DSA=1 torchrun --nnodes 2 --nproc-per-node 8 \
  --node_rank=$rank \
  --master_addr=$master_addr \
  --master_port=$master_port \
  scripts/pretrain.py \
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
  --pretrained_checkpoint "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/fake/align/checkpoints/latest-checkpoint.pt"