#!/bin/bash
#SBATCH --job-name=20m_lora_r256_410m_1.4b
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:3
#SBATCH --output=.slurm_logs/20m_r256_410m_1.4b_lora.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

# accelerate launch src/pretrain_lora.py \
#     --grown_model "models/pythia-70m-to-pythia-410m" \
#     --tokenizer "EleutherAI/pythia-70m" \
#     --dataset "data/train_00/00_20m.pt" \
#     --seed 1234 \
#     --rank 256 \
#     --batch_size 8 \
#     --lr 1e-4 \
#     --output_dir "models/pythia-70m-to-pythia-410m-lora"



accelerate launch src/pretrain_lora.py \
    --grown_model "models/pythia-410m-to-pythia-1.4b" \
    --tokenizer "EleutherAI/pythia-410m" \
    --dataset "data/train_00/00_20m.pt" \
    --seed 1234 \
    --rank 256 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir "models/pythia-410m-to-pythia-1.4b-lora"



    