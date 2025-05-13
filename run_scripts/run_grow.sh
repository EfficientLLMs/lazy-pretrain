#!/bin/bash
#SBATCH --job-name=grow_tiny_olmo
#SBATCH --mem=128G
#SBATCH --output=.slurm_logs/grow_tiny_olmo.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu


python src/grow/grow.py \
    --small_model "pythia-410m" \
    --large_depth 24 \
    --large_width 2048 \
    --depth_growth "alternate" \
    --attn_heads 16 \
    --output_dir "models/pythia-410m-to-pythia-1.4b"
    # --checkpoint_step "step140000"

# python src/grow_olmo/grow.py \
#     --small_model "OLMo-1B" \
#     --large_depth 32 \
#     --large_width 4096 \
#     --depth_growth "alternate" \
#     --output_dir "models/OLMo-1B-to-OLMo-7B" \
#     # --checkpoint_step "step138000"
    
# # "step3000",


# python src/grow_olmo/grow.py \
#     --small_model "models/tiny-olmo-150M-step406934-unsharded" \
#     --large_depth 16 \
#     --large_width 1536 \
#     --depth_growth "alternate" \
#     --output_dir "models/tiny-olmo-150M-to-tiny-olmo-700M" \
    # --checkpoint_step "step138000"
    
# "step3000",