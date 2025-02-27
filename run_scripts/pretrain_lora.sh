#!/bin/bash
#SBATCH --job-name=olmo-5m
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:8
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/olmo-5m.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu
#SBATCH --exclude=shire-1-6,shire-1-10,babel-1-27


accelerate launch src/pretrain/pretrain_lora.py \
    --grown_model "models/OLMo-1B-to-OLMo-7B" \
    --tokenizer "allenai/OLMo-1B" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 2 \
    --lr 1e-5 \
    --output_dir "models/OLMo-1B-to-OLMo-7B-5m-lora-alpha256-allmod-1e-5" \
    --dataset 'dolma' \
    --num_tokens 5_000_000 \
    --chunk_size 2048