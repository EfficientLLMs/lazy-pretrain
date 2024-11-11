#!/bin/bash
#SBATCH --job-name=pythia-70m-step143000-1b-relora
#SBATCH --mem=80G
#SBATCH --gres=gpu:L40S:8
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/pythia-70m-step143000-1b-relora.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuel4@andrew.cmu.edu

export NCCL_P2P_DISABLE=1

accelerate launch src/pretrain/pretrain_relora.py \
    --grown_model "models-xinyue/pythia-70m-step143000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 64 \
    --lr 1e-5 \
    --output_dir "models-xinyue/pythia-70m-step143000-1b-relora" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 1_000_000_000 \
    --chunk_size 512 \
    --scheduler "cosine_restarts" \
    --relora_steps 1413 \
    --cycle_length 1413 \
    --warmup_steps 100 \
    --restart_warmup_steps 50 \
    --min_lr_ratio 0.1
