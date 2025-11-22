#! /bin/bash
cd /mnt/world_foundational_model/gkz/dyva-worldlm
# launch script for multi-node training
export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN="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"
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1

VISION_BACKBONE_TYPE="dyva_dit_siglip"
LLM_BAKCBONE_ID="qwen25-7b-chat" # "llama2-7b-pure"
BATCH_SIZE=8
ALIGN_EPOCHS=1
FINETUNE_EPOCHS=1
# Determine model_id and run_root based on backbone types
if [[ "$LLM_BAKCBONE_ID" == "llama2-7b-pure" ]]; then
    llm=""
elif [[ "$LLM_BAKCBONE_ID" == qwen25-7b-* ]]; then
    llm="_qwen"
else
    echo "Unsupported LLM_BAKCBONE_ID: $LLM_BAKCBONE_ID"
    exit 1
fi

MODEL_TYPE="${VISION_BACKBONE_TYPE}+7b"
MODEL_ID="${VISION_BACKBONE_TYPE}${llm}+7b"
FINETUNE_RUN_ID="${MODEL_ID}_ft_proj_noalign"
IMAGE_RESIZE_STRATEGY="resize-naive" # "anyres_max_9" or "resize-naive"
LLM_MAX_LENGTH=2048
ARCH_SPECIFIER="no-align+dual" # "align+dual" or "align+fused-gelu-mlp"
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

# ========= 多机多卡分布式设置 =========
GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29502"}
NNODES=${WORLD_SIZE:-"2"}       # 这里写总节点数
NODE_RANK=${RANK:-"0"}          # 当前节点 rank

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ========= 训练参数 =========
TRAINING_ARGS=(
    --model.type "$MODEL_TYPE"
    --model.model_id "$MODEL_ID"
    --model.arch_specifier $ARCH_SPECIFIER
    --model.llm_max_length $LLM_MAX_LENGTH
    --model.llm_backbone_id $LLM_BAKCBONE_ID
    --model.image_resize_strategy $IMAGE_RESIZE_STRATEGY
    --model.enable_mixed_precision_training True
    --model.finetune_per_device_batch_size $BATCH_SIZE
    --hf_token HF_TOKEN
    --run_id "$FINETUNE_RUN_ID"
    --run_root_dir "runs/${RUN_ROOT}"
    --dataset.type "$DATASET_TYPE"
    --wandb_project "dyva-worldlm"
    --wandb_entity "$WANDB_ENTITY"
    --stage "finetune"
)

# ========= 启动命令 =========
/mnt/world_foundational_model/gkz/conda/svd/bin/torchrun ${DISTRIBUTED_ARGS[@]} \
    scripts/pretrain.py \
    ${TRAINING_ARGS[@]}
