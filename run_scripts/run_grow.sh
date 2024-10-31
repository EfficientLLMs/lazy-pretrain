#!/bin/bash
#SBATCH --job-name=grow_70m_410m
#SBATCH --mem=32G
#SBATCH --output=.slurm_logs/grow_70m_410m.out

#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu


# python src/grow.py \
#     --small_model "pythia-410m" \
#     --large_depth 24 \
#     --large_width 2048 \
#     --depth_growth "alternate" \
#     --attn_heads 16 \
#     --output_dir "models-xinyue/pythia-410m-step140000-to-pythia-1.4b" \
#     --checkpoint_step "step140000"

python src/grow.py \
    --small_model "pythia-70m" \
    --large_depth 24 \
    --large_width 1024 \
    --depth_growth "alternate" \
    --output_dir "models/pythia-70m-step140000-to-pythia-410m" \
    --checkpoint_step "step140000"
    
# "step3000",