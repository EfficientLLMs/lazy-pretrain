#!/bin/bash
#SBATCH --job-name=create_dolma_tokenized
#SBATCH --mem=32G
#SBATCH --output=.slurm_logs/create_dolma_tokenized.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

python src/prepare_memmap.py \
    --data_dir data/dolma \
    --num_tokens 10_000_000_000 \
    --chunk_size 512 \
    --output_dir data/dolma_tokenized
