#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=2:30:00
#SBATCH --mem=40GB
#SBATCH --output=script_logging/slurm_%A.out

conda create -n wiki_env python=3.8  #-> The version is important. The wikiextractor has problem with more recent version of python
conda activate wiki_env
pip install wikiextractor SoMaJo

python -m wikiextractor.WikiExtractor corpus_datasets/enwiki-20251001-pages-articles.xml.bz2 -o corpus_datasets/enwiki-20251001 --processes 16 --no-templates

