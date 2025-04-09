#!/bin/bash
#SBATCH --job-name=olmo-5b
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:6
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/olmo-5b.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu
#SBATCH --exclude=shire-1-6,shire-1-10,babel-1-27


accelerate launch src/pretrain/pretrain_lora.py \
    --grown_model "models/OLMo-1B-to-OLMo-7B" \
    --tokenizer "allenai/OLMo-1B" \
    --seed 1234 \
    --rank 8 \
    --lora_alpha 8 \
    --batch_size 1 \
    --lr 1e-5 \
    --output_dir "models/OLMo-1B-to-OLMo-7B-5b-lora-alpha256-allmod-1e-5" \
    --dataset 'dolma' \
    --num_tokens 5_000_000_000 \
    --chunk_size 2048