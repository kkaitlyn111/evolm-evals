#!/bin/bash
#SBATCH --job-name=math_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --array=0-23  # Adjust based on number of model-dataset combinations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Create logs directory if it doesn't exist
mkdir -p logs

# Define models and datasets
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-Math-7B-Instruct"
)

DATASETS=(
    "GSM8KPlatinum"
    "MATH"
    "AIME2023" 
    "AIME2024"
    "MATHLevel3"
    "MinervaMath"
)

# Calculate model and dataset indices
NUM_DATASETS=${#DATASETS[@]}
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_DATASETS))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % NUM_DATASETS))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

echo "Running evaluation:"
echo "  Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  GPU: $CUDA_VISIBLE_DEVICES"

# Run the evaluation
cd /path/to/evolm  # Change this to your project path

python evaluation/simple_eval.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --max_new_tokens 1024 \
    --k 10 \
    --device cuda \
    --output_dir "evaluation/results/batch_$(date +%Y%m%d)"

echo "Evaluation complete for $MODEL on $DATASET" 