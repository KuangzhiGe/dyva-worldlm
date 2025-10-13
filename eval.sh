export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN="YOUR_HF_KEY"
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
cd /mnt/world_foundational_model/gkz/prismatic-vlms/vlm-evaluation
data_dir="/mnt/world_foundational_model/gkz/data/vlm-evaluation"

model_id="prism-svd_dual_siglip"
run_root=$(get_run_root "$model_id")
finetune_run_id="${model_id}_ft_proj_noalign"
model_dir="/mnt/world_foundational_model/gkz/prismatic-vlms/runs/${run_root}/${finetune_run_id}"
nproc_per_node=8

accelerate launch scripts/evaluate.py \
    --model_family prismatic \
    --model_id "${model_id}" \
    --dataset_root_dir "$data_dir" \
    --results_dir "${model_dir}/results_true_multi" \
    --model_dir "$model_dir" \
    --dataset_types ["sat-full"]

python scripts/score.py --model_id "${model_id}" --dataset_root_dir "$data_dir" --results_dir "${model_dir}/results_true_multi" --dataset_types ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]