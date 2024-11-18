#!/bin/bash
#SBATCH --job-name=pythia-70m-step141000-4b-relora-new
#SBATCH --mem=80G
#SBATCH --gres=gpu:A6000:8
#SBATCH --exclude=babel-1-23
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/pythia-70m-step141000-4b-relora-new.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuel4@andrew.cmu.edu

python src/pretrain/pretrain_relora.py \
    --grown_model "models-xinyue/pythia-70m-step141000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 64 \
    --lr 1e-5 \
    --output_dir "models-xinyue/pythia-70m-step141000-4b-relora-new" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 4_000_000_000 \
    --chunk_size 512 \
    --scheduler "cosine_restarts" \
    --min_lr_ratio 0.1 \
    --wandb_entity "irisiris" \
    --num_restarts 10 \
    --do_extact_lora
    # we don't need others; but we need to specify the num_restarts in the main code
    # --relora_steps 350 \
    # --cycle_length 350 \
    # --warmup_steps 50 \
    # --restart_warmup_steps 50 \