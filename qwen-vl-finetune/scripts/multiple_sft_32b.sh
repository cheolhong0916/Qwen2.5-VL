#!/bin/bash

# =================================================================
# Sequential Finetuning Script
# =================================================================
# This script fine-tunes the baseline model on three different
# datasets sequentially: synthetic, dynamic, and perception.
# Each training run will create a separate output directory.
# =================================================================

# --- Common Configuration ---

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-32B-Instruct

# Training hyperparameters
lr=2e-7
batch_size=2
grad_accum_steps=8

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# --- Sequential Training Loop ---

# Define the datasets to train on, in order
DATASET_LIST=("synthetic" "dynamic" "perception")

for dataset_name in "${DATASET_LIST[@]}"; do
    # Set a unique port for each run to avoid conflicts
    MASTER_PORT=$(shuf -i 20001-29999 -n 1)

    echo "================================================================"
    echo "===== Starting training for dataset: ${dataset_name} ====="
    echo "================================================================"

    # Output configuration for the current dataset
    run_name="qwen2vl-32B-baseline-${dataset_name}"
    output_dir=./output/${run_name}

    # Training arguments
    # Note: We construct the full command here for clarity
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

    # Launch training for the current dataset
    torchrun --nproc_per_node=${NPROC_PER_NODE} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             ${entry_file} ${args}

    # Check if the training was successful before proceeding
    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!!!! Training for dataset ${dataset_name} failed. Halting script. !!!!!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
done

echo "================================================================"
echo "===== All training runs completed successfully. ====="
echo "================================================================"