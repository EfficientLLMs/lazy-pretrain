#!/bin/bash
#SBATCH --job-name=test-preempt
#SBATCH --mem=80G
#SBATCH --gres=gpu:A6000:8
#SBATCH --exclude=babel-0-37,babel-0-31
#SBATCH --partition=preempt
#SBATCH --output=.slurm_logs/test-preempt.out
#SBATCH --time=00-00:5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuel4@andrew.cmu.edu

export NCCL_P2P_DISABLE=1

accelerate launch src/pretrain/pretrain_relora_preempt.py \
    --grown_model "models-xinyue/pythia-70m-step141000-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 64 \
    --lr 1e-5 \
    --output_dir "models-xinyue/test-preempt" \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 4_000_000_000 \
    --chunk_size 512 \
    --scheduler "cosine_restarts" \
    --min_lr_ratio 0.1 \
    --wandb_entity "irisiris" \
    --num_restarts 20 \