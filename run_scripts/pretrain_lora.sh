#!/bin/bash
#SBATCH --job-name=pythia-70m-step142000-1b-lora-alpha256-allmod-1e-5
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:8
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/pythia-70m-step142000-1b-lora-alpha256-allmod-1e-5.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu
#SBATCH --exclude=shire-1-6,shire-1-10,babel-1-27


accelerate launch src/pretrain/pretrain_lora.py \
    --grown_model "models/pythia-70m-step142000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 64 \
    --lr 1e-5 \
    --output_dir "models/pythia-70m-step142000-1b-lora-alpha256-allmod-1e-5" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 1_000_000_000 \
    --chunk_size 512