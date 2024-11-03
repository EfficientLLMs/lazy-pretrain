#!/bin/bash
#SBATCH --job-name=100m_lora_r256_70m_410m_alpha256-allmod
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:8
#SBATCH --output=.slurm_logs/1b_r256_70m_410m_lora_alpha256-allmod.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuel4@andrew.cmu.edu

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
#     --batch_size 8 \
#     --lr 1e-4 \
#     --output_dir "models/pythia-70m-to-pythia-410m-lora-"


# --grown_model "models/pythia-410m-to-pythia-1.4b" \
# --dataset "data/train_00/00_20m.pt" \

accelerate launch src/pretrain_lora.py \
    --grown_model "models/pythia-70m-step140000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 64 \
    --lr 1e-5 \
    --output_dir "models/pythia-70m-step140000-to-pythia-410m-lora-alpha256-allmod" \
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

accelerate launch src/pretrain_lora.py \
    --grown_model "models-xinyue/pythia-70m-step140000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 64 \
    --lr 1e-3 \
    --output_dir "models-xinyue/pythia-70m-step140000-to-pythia-410m-lora-alpha256-allmod-1e-3" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 1_000_000_000 \
    --chunk_size 512