#!/bin/bash
#SBATCH --job-name=grow_410_1_4b
#SBATCH --mem=32G
#SBATCH --output=.slurm_logs/grow_410_1_4b.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu


python src/grow.py \
    --small_model "pythia-410m" \
    --large_depth 24 \
    --large_width 2048 \
    --depth_growth "alternate" \
    --attn_heads 16 \
    --output_dir "models/pythia-410m-to-pythia-1.4b"

# python src/grow.py \
#     --small_model "pythia-70m" \
#     --large_depth 24 \
#     --large_width 1024 \
#     --depth_growth "alternate" \
#     --output_dir "models/pythia-70m-to-pythia-410m"
    
# "step3000",