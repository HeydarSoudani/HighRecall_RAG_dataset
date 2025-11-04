#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --time=1:00:00
#SBATCH --mem=20GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
model_name_or_path="Zill1/StepSearch-7B-Base"
generation_model="step_search"
retriever_name="rerank_l6"
run="run_1"


# accelerate launch --multi_gpu
srun python $HOME/HighRecall_DS/c2_model_generation/model_inference.py \
    --model_name_or_path "$model_name_or_path" \
    --generation_model "$generation_model" \
    --retriever_name "$retriever_name" \
    --run "$run"
