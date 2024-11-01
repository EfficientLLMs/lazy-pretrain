#!/bin/bash
#SBATCH --job-name=1b_lora_r256_70m_410m_r8_alpha8-allmod-1e-3
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:4
#SBATCH -p preempt
#SBATCH --output=.slurm_logs/1b_r256_70m_410m_lora_r8_alpha8-allmod-1e-3.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu



accelerate launch src/pretrain/pretrain_lora.py \
    --grown_model "models/pythia-70m-step140000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 8 \
    --lora_alpha 8 \
    --batch_size 64 \
    --lr 1e-3 \
    --output_dir "models/pythia-70m-step140000-to-pythia-410m-lora-r8-alpha8-allmod-1e-3" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 1_000_000_000 \
    --chunk_size 512

# accelerate launch src/pretrain_lora.py \
#     --grown_model "models-xinyue/pythia-70m-step140000-to-pythia-410m" \
#     --tokenizer "EleutherAI/pythia-70m" \
#     --seed 1234 \
#     --rank 256 \
#     --batch_size 32 \
#     --lr 1e-4 \
#     --output_dir "models-xinyue/pythia-70m-step140000-to-pythia-410m-lora" \
#     --use_on_the_fly \
#     --first_idx 19 \
#     --last_idx 20 \
#     --num_tokens 1000000000 \
#     --chunk_size 512

    