#!/bin/bash
#SBATCH --job-name=100m_freeze_70m_410m
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:2
#SBATCH --output=.slurm_logs/100m_freeze_70m_410m.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu


accelerate launch src/pretrain_freeze.py \
    --grown_model "models/pythia-70m-to-pythia-410m" \
    --tokenizer "EleutherAI/pythia-70m" \
    --dataset "data/train_00/00_100m.pt" \
    --seed 1234 \
    --batch_size 8 \
    --lr 1e-4 \
    --output_dir "models/pythia-70m-to-pythia-410m-freeze-100m"
    # --eval_results_path "eval/freeze_70m_410m_eval_results_20m" \
    # --parallelize \
    # --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \
    # --output_dir "models/pythia-70m-to-pythia-410m-freeze-20m"


# accelerate launch src/pretrain_lora.py \
#     --grown_model "models/pythia-70m-to-pythia-410m" \
#     --tokenizer "EleutherAI/pythia-70m" \
#     --dataset "data/train_00/00_100m.pt" \
#     --seed 1234 \
#     --rank 256 \
#     --batch_size 8 \
#     --lr 1e-4 \
#     --output_dir "models/pythia-70m-to-pythia-410m-lora-20m"



# accelerate launch src/pretrain_lora.py \
#     --grown_model "models/pythia-410m-to-pythia-1.4b" \
#     --tokenizer "EleutherAI/pythia-410m" \
#     --dataset "data/train_00/00_20m.pt" \
#     --seed 1234 \
#     --rank 256 \
#     --batch_size 8 \
#     --lr 1e-4 \
#     --output_dir "models/pythia-410m-to-pythia-1.4b-lora"



    