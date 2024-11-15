#!/bin/bash
#SBATCH --job-name=pythia-70m-step142000-2b-relora
#SBATCH --mem=80G
#SBATCH --gres=gpu:A6000:8
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/pythia-70m-step142000-2b-relora.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu
#SBATCH --exclude=shire-1-6,shire-1-10,babel-1-27,babel-1-23

# export NCCL_P2P_DISABLE=1

accelerate launch src/pretrain/pretrain_relora.py \
    --grown_model "models/pythia-70m-step142000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 64 \
    --lr 1e-5 \
    --output_dir "models/pythia-70m-step142000-2b-relora" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 2_000_000_000 \
    --chunk_size 512 \
    --scheduler "cosine_restarts" \
    --min_lr_ratio 0.1 \
    --wandb_entity "vibhamasti" \
    --num_restarts 10

    # we don't need others; but we need to specify the num_restarts in the main code
    # --relora_steps 350 \
    # --cycle_length 350 \
    # --warmup_steps 50 \
    # --restart_warmup_steps 50 \
