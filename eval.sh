export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN="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"
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=8

VISION_BACKBONE_TYPE="dyva_cogvideox"
LLM_BAKCBONE_ID="qwen25-7b-chat" # "llama2-7b-pure"
# Determine model_id and run_root based on backbone types
if [[ "$LLM_BAKCBONE_ID" == "llama2-7b-pure" ]]; then
    llm=""
elif [[ "$LLM_BAKCBONE_ID" == qwen25-7b-* ]]; then
    llm="_qwen"
else
    echo "Unsupported LLM_BAKCBONE_ID: $LLM_BAKCBONE_ID"
    exit 1
fi

MODEL_ID="${VISION_BACKBONE_TYPE}${llm}+7b"
FINETUNE_RUN_ID="${MODEL_ID}_ft_proj_noalign"
IMAGE_RESIZE_STRATEGY="resize-naive" # "anyres_max_9" or "resize-naive"
LLM_MAX_LENGTH=2048
ARCH_SPECIFIER="no-align+fused-gelu-mlp" # "align+dual" or "align+fused-gelu-mlp"
WANDB_ENTITY="2200013209-peking-university"
DATASET_TYPE="llava-svd" #"llava-svd_multi"

if [[ "${VISION_BACKBONE_TYPE}" == "svd" || "${VISION_BACKBONE_TYPE}" == "svd_unet" ]]; then
    RUN_ROOT="svd_only"
elif [[ "${VISION_BACKBONE_TYPE}" == svd_* || "${VISION_BACKBONE_TYPE}" == svd_dual_* || "${VISION_BACKBONE_TYPE}" == dyva_* ]]; then
    RUN_ROOT="${VISION_BACKBONE_TYPE/svd_dual_/svd_/dyva_}"
else
    RUN_ROOT="${VISION_BACKBONE_TYPE}"
    echo "Warning: vision_backbone_type不符合预设规则，run_root已设置为'${RUN_ROOT}'"
fi
echo "RUN_ROOT = $RUN_ROOT"

# Evaluation
# "refcoco-slim","ocid-ref-slim",
cd /mnt/world_foundational_model/gkz/prismatic-vlms/vlm-evaluation
data_dir="/mnt/world_foundational_model/gkz/data/vlm-evaluation"
#,"mmsi-full"
# ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]
model_dir="/mnt/world_foundational_model/gkz/dyva-worldlm/runs/${RUN_ROOT}/${FINETUNE_RUN_ID}"

accelerate launch \
    --num_processes=$nproc_per_node \
    scripts/evaluate.py \
    --model_family prismatic \
    --model_id "${MODEL_ID}" \
    --dataset_root_dir "$data_dir" \
    --results_dir "${model_dir}/res" \
    --model_dir "$model_dir" \
    --dataset_types ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]

python scripts/score.py --model_id "${MODEL_ID}" --dataset_root_dir "$data_dir" --results_dir "${model_dir}/res" --dataset_types ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]
# ["sat-full","seed-bench","vqa-v2-slim","gqa-slim","vizwiz-slim","vsr-full","pope-slim","tally-qa-slim","3dsr-full","spatial-mm-obj","mmsi-full","mindcube-full","spar-full"]



# FAKE TRAINING
cd /mnt/world_foundational_model/gkz/prismatic-vlms
TORCH_USE_CUDA_DSA=1 torchrun --nnodes 1 --nproc-per-node $nproc_per_node scripts/pretrain.py \
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
    # --model.align_learning_rate 1e-6 \
    # --model.align_max_grad_norm 0.5 \
    # --pretrained_checkpoint "/mnt/world_foundational_model/gkz/ckpts/hub/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/prism-dinosiglip+7b" \