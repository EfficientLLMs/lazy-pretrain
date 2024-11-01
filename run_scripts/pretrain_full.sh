#!/bin/bash
#SBATCH --job-name=1b_full_70m_410m_step140000-1e-3
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:8
#SBATCH --output=.slurm_logs/1b_full_70m_410m_step140000-1e-3.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu



accelerate launch src/pretrain/pretrain_full.py \
    --grown_model "models/pythia-70m-step140000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --batch_size 8 \
    --lr 1e-3 \
    --output_dir "models/pythia-70m-step140000-to-pythia-410m-full-1e-3" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 1_000_000_000 \
    --chunk_size 512
    