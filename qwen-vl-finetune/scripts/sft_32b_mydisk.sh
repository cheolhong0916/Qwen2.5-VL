#!/bin/bash

# =================================================================
# Sequential Finetuning Script for Large Models (e.g., 32B)
# =================================================================
# This script fine-tunes a single large model sequentially on a list
# of datasets. Each training job utilizes all available GPUs.
# =================================================================

# --- Configuration ---

# To see the exact commands being run, uncomment the line below.
# set -x

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# DeepSpeed configuration
deepspeed=./scripts/zero3_offload_pin_memory_false.json

# Model configuration
llm=Qwen/Qwen2.5-VL-32B-Instruct

# Training hyperparameters
lr=2e-7
batch_size=2
grad_accum_steps=8

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# List of datasets to train on sequentially
# DATASET_LIST=("real" "static" "reasoning" "2d" "3d" "synthetic" "dynamic" "perception")
DATASET_LIST=("static" "reasoning" "2d" "3d" "synthetic" "dynamic" "perception")

# Base directory for all outputs
OUTPUT_BASE_DIR="/data/shared/Qwen/mydisk"

# --- Execution ---

echo "================================================================"
echo "Starting sequential fine-tuning for ${#DATASET_LIST[@]} datasets."
echo "Model: ${llm}"
echo "GPUs to be used per run: ${NPROC_PER_NODE}"
echo "================================================================"

# Get a single timestamp for this entire batch of training runs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Loop through each dataset and run training sequentially
for dataset_name in "${DATASET_LIST[@]}"; do
    echo ""
    echo "----------------------------------------------------------------"
    echo "--- Starting training for dataset: ${dataset_name} ---"
    echo "----------------------------------------------------------------"

    # Set a unique master port for each run to avoid conflicts
    MASTER_PORT=$(shuf -i 20001-29999 -n 1)

    # Define unique names and directories for this specific run
    run_name="${llm}-${dataset_name}-${TIMESTAMP}"
    output_dir=${OUTPUT_BASE_DIR}/output/${run_name}

    echo "Run Name: ${run_name}"
    echo "Output will be saved to: ${output_dir}"

    # Training arguments
    # Note: We construct the args string inside the loop to use the unique variables
    args="
        --deepspeed ${deepspeed} \
        --model_name_or_path "${llm}" \
        --dataset_use ${dataset_name} \
        --data_flatten False \
        --tune_mm_vision False \
        --tune_mm_mlp True \
        --tune_mm_llm True \
        --bf16 \
        --output_dir ${output_dir} \
        --num_train_epochs 0.5 \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size $((batch_size*2)) \
        --gradient_accumulation_steps ${grad_accum_steps} \
        --max_pixels 50176 \
        --min_pixels 784 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 1 \
        --learning_rate ${lr} \
        --weight_decay 0 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 8192 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --run_name ${run_name} \
        --report_to wandb"

    # Launch the training job using all available GPUs
    torchrun --nproc_per_node=${NPROC_PER_NODE} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             ${entry_file} ${args}

    # Check the exit code of the last command.
    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!!!! Training for dataset ${dataset_name} failed. Halting script. !!!!!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi

    echo "--- ✅ Successfully completed training for dataset: ${dataset_name} ---"
done

echo ""
echo "================================================================"
echo "🎉 All sequential training runs completed successfully! 🎉"
echo "================================================================"