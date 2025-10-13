export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN="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"
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=8
nnodes_num=2
total_processes=$((nproc_per_node * nnodes_num))

get_run_root() {
    local model_id="$1"
    local run_root
    vb_type="${model_id#prism-}"
    if [[ "${vb_type}" == "svd" || "${vb_type}" == "svd_unet" ]]; then
        run_root="svd_only"
    elif [[ "${vb_type}" == svd_* || "${vb_type}" == svd_dual_* ]]; then
        run_root="${vb_type/svd_dual_/svd_}"
    else
        run_root="baseline"
        echo "Warning: vision_backbone_type不符合预设规则，run_root已设置为'${run_root}'" >&2
    fi

    echo "$run_root"
}

# Evaluation
# "refcoco-slim","ocid-ref-slim",
cd /mnt/world_foundational_model/gkz/prismatic-vlms/vlm-evaluation
data_dir="/mnt/world_foundational_model/gkz/data/vlm-evaluation"
#,"mmsi-full"
# ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]

master_addr=10.40.0.81
master_port=12138
rank=0

model_id="prism-svd_dual_siglip"
run_root=$(get_run_root "$model_id")
finetune_run_id="${model_id}_ft_proj_noalign"
model_dir="/mnt/world_foundational_model/gkz/prismatic-vlms/runs/${run_root}/${finetune_run_id}"

# accelerate launch \
#     --num_machines=$nnodes_num \
#     --num_processes=$total_processes \
#     --main_process_ip=$master_addr \
#     --main_process_port=$master_port \
#     --machine_rank=$rank \
#     scripts/evaluate.py \
#     --model_family prismatic \
#     --model_id "${model_id}" \
#     --dataset_root_dir "$data_dir" \
#     --results_dir "${model_dir}/results_true_multi" \
#     --model_dir "$model_dir" \
#     --dataset_types ["sat-full"]

python scripts/score.py --model_id "${model_id}" --dataset_root_dir "$data_dir" --results_dir "${model_dir}/results_true_multi" --dataset_types ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]


# FAKE TRAINING
# cd /mnt/world_foundational_model/gkz/prismatic-vlms

# TORCH_USE_CUDA_DSA=1 torchrun --nnodes 2 --nproc-per-node 8 \
#     --node_rank=$rank \
#     --master_addr=$master_addr \
#     --master_port=$master_port \
#     scripts/pretrain.py \
#     --model.type "prism-svd" \
#     --model.model_id "prism-svd" \
#     --model.vision_backbone_id "clip-svd" \
#     --model.image_resize_strategy "resize-naive" \
#     --model.llm_backbone_id "llama2-7b-pure" \
#     --model.arch_specifier "align+gelu-mlp" \
#     --model.enable_mixed_precision_training True \
#     --model.finetune_per_device_batch_size 2 \
#     --model.finetune_epochs 100000 \
#     --dataset.type "llava-svd" \
#     --hf_token HF_TOKEN \
#     --run_id "finetune_fake" \
#     --wandb_project "svd_vlm" \
#     --wandb_entity "2200013209-peking-university" \
#     --stage "finetune" \
#     --pretrained_checkpoint "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/fake/align/checkpoints/latest-checkpoint.pt"