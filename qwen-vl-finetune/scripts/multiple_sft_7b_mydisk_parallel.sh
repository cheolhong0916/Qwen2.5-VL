#!/bin/bash

# =================================================================
# Parallel Finetuning Script (2 Models Simultaneously)
# =================================================================
# This script fine-tunes the baseline model on multiple datasets
# with 2 models running in parallel on A100 GPUs.
# GPU allocation: Model 1 uses GPUs 0-3, Model 2 uses GPUs 4-7
# =================================================================

# --- Common Configuration ---

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NNODES=${WORLD_SIZE:-1}
TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
NPROC_PER_NODE=$((TOTAL_GPUS / 2))  # 4 GPUs per model (assuming 8 total)

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct
# llm=Qwen/Qwen2-VL-7B-Instruct
# llm=Qwen/Qwen2-VL-7B

# Training hyperparameters
lr=2e-7
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Output base directory - MODIFIED TO USE EXTERNAL STORAGE
OUTPUT_BASE_DIR="/data/shared/Qwen/mydisk"

# Optional: Add timestamp for unique runs (uncomment if needed)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create base output directory if it doesn't exist
echo "Creating output base directory: ${OUTPUT_BASE_DIR}"
mkdir -p ${OUTPUT_BASE_DIR}/output

echo "================================================================"
echo "GPU Configuration:"
echo "Total GPUs: ${TOTAL_GPUS}"
echo "GPUs per model: ${NPROC_PER_NODE}"
echo "Running 2 models in parallel"
echo "================================================================"

# --- Function to run training ---
run_training() {
    local dataset_name=$1
    local gpu_group=$2  # 0 for GPUs 0-3, 1 for GPUs 4-7
    local model_id=$3   # 1 or 2
    
    # Set GPU visibility based on group
    if [ $gpu_group -eq 0 ]; then
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        local port_base=20000
    else
        export CUDA_VISIBLE_DEVICES="4,5,6,7"
        local port_base=25000
    fi
    
    # Set a unique port for each model
    MASTER_PORT=$((port_base + RANDOM % 1000))

    echo "[Model ${model_id}] Starting training for dataset: ${dataset_name}"
    echo "[Model ${model_id}] Using GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "[Model ${model_id}] Using port: ${MASTER_PORT}"

    # Output configuration for the current dataset
    run_name="${llm}-${dataset_name}-${TIMESTAMP}"
    output_dir=${OUTPUT_BASE_DIR}/output/${run_name}
    
    # Create specific output directory for this run
    echo "[Model ${model_id}] Creating output directory: ${output_dir}"
    mkdir -p ${output_dir}

    # Training arguments
    args="
        --deepspeed ${deepspeed} \
        --model_name_or_path ${llm} \
        --dataset_use ${dataset_name} \
        --data_flatten False \
        --tune_mm_vision False \
        --tune_mm_mlp True \
        --tune_mm_llm True \
        --bf16 \
        --output_dir ${output_dir} \
        --num_train_epochs 1 \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size $((batch_size*2)) \
        --gradient_accumulation_steps ${grad_accum_steps} \
        --max_pixels 50176 \
        --min_pixels 784 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 1000 \
        --save_total_limit 1 \
        --learning_rate ${lr} \
        --weight_decay 0 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 8192 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --run_name ${run_name} \
        --report_to wandb"

    # Launch training for the current dataset
    echo "[Model ${model_id}] Starting torchrun for ${dataset_name}..."
    torchrun --nproc_per_node=${NPROC_PER_NODE} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             ${entry_file} ${args}

    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "[Model ${model_id}] ❌ Training for dataset ${dataset_name} failed with exit code ${exit_code}"
        return $exit_code
    else
        echo "[Model ${model_id}] ✅ Training for dataset ${dataset_name} completed successfully"
        echo "[Model ${model_id}] Output saved to: ${output_dir}"
        return 0
    fi
}

# --- Parallel Training Loop ---

# Define the datasets to train on, in order

# Fine tuning with Fine-tuning-data
DATASET_LIST=("real" "static" "reasoning" "2d" "3d" "synthetic" "dynamic" "perception")

# Fine tuning with Fine-tuning-data + PIXMO (point predictinon dataset)
# DATASET_LIST=("synthetic_pixmo" "real_pixmo" "static_pixmo" "dynamic_pixmo" "perception_pixmo" "reasoning_pixmo" "2d_pixmo" "3d_pixmo")

# Group datasets into pairs for parallel execution
DATASET_PAIRS=()
for ((i=0; i<${#DATASET_LIST[@]}; i+=2)); do
    if [ $((i+1)) -lt ${#DATASET_LIST[@]} ]; then
        # Pair of datasets
        DATASET_PAIRS+=("${DATASET_LIST[i]},${DATASET_LIST[i+1]}")
    else
        # Single dataset (if odd number)
        DATASET_PAIRS+=("${DATASET_LIST[i]}")
    fi
done

echo "================================================================"
echo "Training Plan:"
for ((i=0; i<${#DATASET_PAIRS[@]}; i++)); do
    IFS=',' read -ra PAIR <<< "${DATASET_PAIRS[i]}"
    if [ ${#PAIR[@]} -eq 2 ]; then
        echo "  Batch $((i+1)): ${PAIR[0]} + ${PAIR[1]} (parallel)"
    else
        echo "  Batch $((i+1)): ${PAIR[0]} (single)"
    fi
done
echo "================================================================"

# Execute training in parallel batches
for ((batch=0; batch<${#DATASET_PAIRS[@]}; batch++)); do
    IFS=',' read -ra CURRENT_PAIR <<< "${DATASET_PAIRS[batch]}"
    
    echo ""
    echo "================================================================"
    echo "===== BATCH $((batch+1))/${#DATASET_PAIRS[@]} ====="
    
    if [ ${#CURRENT_PAIR[@]} -eq 2 ]; then
        echo "===== Running ${CURRENT_PAIR[0]} and ${CURRENT_PAIR[1]} in parallel ====="
        echo "================================================================"
        
        # Start both trainings in background
        (run_training "${CURRENT_PAIR[0]}" 0 1) &
        PID1=$!
        
        (run_training "${CURRENT_PAIR[1]}" 1 2) &
        PID2=$!
        
        # Wait for both to complete
        echo "Waiting for parallel training to complete..."
        echo "  Model 1 (${CURRENT_PAIR[0]}) PID: $PID1"
        echo "  Model 2 (${CURRENT_PAIR[1]}) PID: $PID2"
        
        wait $PID1
        EXIT1=$?
        wait $PID2
        EXIT2=$?
        
        # Check results
        if [ $EXIT1 -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!!!!! Model 1 training (${CURRENT_PAIR[0]}) failed. Halting script. !!!!!"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # Kill the other process if still running
            kill $PID2 2>/dev/null
            exit 1
        fi
        
        if [ $EXIT2 -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!!!!! Model 2 training (${CURRENT_PAIR[1]}) failed. Halting script. !!!!!"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            exit 1
        fi
        
        echo "✅ Batch $((batch+1)) completed successfully!"
        echo "  ✅ ${CURRENT_PAIR[0]} training completed"
        echo "  ✅ ${CURRENT_PAIR[1]} training completed"
        
    else
        # Single dataset (odd number case)
        echo "===== Running ${CURRENT_PAIR[0]} (single model) ====="
        echo "================================================================"
        
        run_training "${CURRENT_PAIR[0]}" 0 1
        
        if [ $? -ne 0 ]; then
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!!!!! Training for dataset ${CURRENT_PAIR[0]} failed. Halting script. !!!!!"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            exit 1
        fi
        
        echo "✅ ${CURRENT_PAIR[0]} training completed successfully"
    fi
    
    echo ""
done

echo "================================================================"
echo "===== All training runs completed successfully! ====="
echo "================================================================"
echo "All outputs have been saved to: ${OUTPUT_BASE_DIR}/output/"
echo ""
echo "Final output directories:"
for dataset_name in "${DATASET_LIST[@]}"; do
    run_name="${llm}-${dataset_name}-${TIMESTAMP}"
    echo "  - ${dataset_name}: ${OUTPUT_BASE_DIR}/output/${run_name}"
done

echo ""
echo "Training Summary:"
echo "  Total datasets: ${#DATASET_LIST[@]}"
echo "  Parallel batches: ${#DATASET_PAIRS[@]}"
echo "  Timestamp: ${TIMESTAMP}"
echo "  GPU allocation: 4 GPUs per model (0-3, 4-7)"