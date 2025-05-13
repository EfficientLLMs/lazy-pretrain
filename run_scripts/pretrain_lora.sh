#!/bin/bash
#SBATCH --job-name=pythia-410-1.4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:6
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/pythia-410-1.4.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu
#SBATCH --exclude=shire-1-6,shire-1-10,babel-1-27,babel-0-37


# accelerate launch src/pretrain/pretrain_lora.py \
#     --grown_model "models/OLMo-1B-to-OLMo-7B" \
#     --tokenizer "allenai/OLMo-1B" \
#     --seed 1234 \
#     --rank 256 \
#     --lora_alpha 256 \
#     --batch_size 4 \
#     --lr 1e-5 \
#     --output_dir "models/OLMo-1B-to-OLMo-7B-5b-lora-alpha256-atn_proj-1e-5" \
#     --dataset 'dolma' \
#     --num_tokens 5_000_000_000 \
#     --chunk_size 2048


accelerate launch src/pretrain/pretrain_lora.py \
    --grown_model "models/pythia-410m-to-pythia-1.4b" \
    --tokenizer "EleutherAI/pythia-70m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 32 \
    --lr 1e-5 \
    --output_dir "models/pythia-410m-to-pythia-1.4b-lora-5b" \
    --dataset 'pile' \
    --num_tokens 5_000_000_000 \
    --chunk_size 1024 \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --wandb_entity vibhamasti \
    --wandb_run_name "pythia-410-1.4b-5b-tokens"