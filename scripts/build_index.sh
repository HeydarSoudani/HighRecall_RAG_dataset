#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --time=5:30:00
#SBATCH --mem=60GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

### === Set variables ==========================
corpus_file=corpus_datasets/enwiki_20251001.jsonl
save_dir=/projects/0/prjs0834/heydars/INDICES
retriever_name=contriever

srun python $HOME/HighRecall_DS/c2_model_generation/src/index_builder.py \
    --retrieval_method $retriever_name \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16
    # --save_embedding \
    