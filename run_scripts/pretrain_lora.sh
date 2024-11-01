#!/bin/bash
#SBATCH --job-name=1b_lora_r64_70m-step140000_410m_1e5
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:8
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/1b_lora_r64_70m-step140000_410m_1e5.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuel4@andrew.cmu.edu

accelerate launch src/pretrain_lora.py \
    --grown_model "models-xinyue/pythia-70m-step140000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-70m" \
    --seed 1234 \
    --rank 64 \
    --batch_size 32 \
    --lr 1e-5 \
    --output_dir "models-xinyue/pythia-70m-step140000-to-pythia-410m-1b-lora-1e5" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 1000000000 \
    --chunk_size 512


# accelerate launch src/pretrain_lora.py \
#     --grown_model "models/pythia-70m-to-pythia-410m" \
#     --tokenizer "EleutherAI/pythia-70m" \
#     --dataset "data/train_00/00_20m.pt" \
#     --seed 1234 \
#     --rank 256 \
#     --batch_size 8 \
#     --lr 1e-4 \
#     --output_dir "models/pythia-70m-to-pythia-410m-lora"



# accelerate launch src/pretrain_lora.py \
#     --grown_model "models/pythia-410m-to-pythia-1.4b" \
#     --tokenizer "EleutherAI/pythia-410m" \
#     --dataset "data/train_00/00_20m.pt" \
#     --seed 1234 \
#     --rank 256 \
#     --batch_size 16 \
#     --lr 1e-4 \
#     --output_dir "models/pythia-410m-to-pythia-1.4b-lora"

    