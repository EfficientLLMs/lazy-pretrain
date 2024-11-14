#!/bin/bash
#SBATCH --job-name=eval_6b_full-1e-5-pythia-70m-step141000-to-pythia-410m
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:2
#SBATCH --output=.slurm_logs/eval_6b_full-1e-5-pythia-70m-step141000-to-pythia-410m.out
#SBATCH --time=01-00:00
#SBATCH --exclude=shire-1-6,shire-1-10,babel-1-27,babel-1-31,babel-0-37
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

# Define variables
EXP_NAME="6b_full-1e-5"
STEP="step140000"
MODEL_NAME="pythia-70m-"$STEP"-to-pythia-410m"

GROWN_MODEL="models/"$MODEL_NAME
TRAINED_MODEL="models/"$MODEL_NAME"-"$EXP_NAME

python src/eval_llm.py \
    --base_model_path $TRAINED_MODEL \
    --tokenizer_path "EleutherAI/pythia-70m" \
    --eval_results_path "eval/eval_"$EXP_NAME"-"$MODEL_NAME \
    --tasks "paloma" "lambada_openai" \
    --token ".token"

# --lora_path "models/pythia-70m-step140000-to-pythia-410m-lora-alpha256-allmod" \
# pythia-70m-step142000-to-pythia-410m-full-1b-1e-5

# eval_full_70m_step142000_410m_eval_results_1b_1e-5

# "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \
# python src/eval_llm.py \
#     --base_model_path "models/pythia-70m-step142000-to-pythia-410m-full-2b-1e-5" \
#     --tokenizer_path "EleutherAI/pythia-70m" \
#     --eval_results_path "eval/eval_full_70m_step142000_410m_eval_results_2b_1e-5" \
#     --tasks "paloma" "lambada_openai" \
#     --token ".token"


# python src/eval_llm.py \
#     --base_model_path "models/pythia-70m-to-pythia-410m" \
#     --lora_path "models/pythia-70m-to-pythia-410m-lora-20m" \
#     --tokenizer_path "EleutherAI/pythia-70m" \
#     --eval_results_path "eval/r256_70m_410m_eval_results_20m" \
#     --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \

# python src/eval_llm.py \
#     --base_model_path "EleutherAI/pythia-70m" \
#     --tokenizer_path "EleutherAI/pythia-70m" \
#     --eval_results_path "eval/70m_eval_results" \
#     --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \

# python src/eval_llm.py \
#     --base_model_path "EleutherAI/pythia-410m" \
#     --checkpoint_step "step140000" \
#     --tokenizer_path "EleutherAI/pythia-410m" \
#     --eval_results_path "eval/410m_step140000_eval_results" \
#     --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \

# python src/eval_llm.py \
#     --base_model_path "models-xinyue/pythia-70m-step140000-to-pythia-410m" \
#     --tokenizer_path "EleutherAI/pythia-70m" \
#     --eval_results_path "eval/70m_step140000_410m_eval_results" \
#     --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \

# python src/eval_llm.py \
#     --base_model_path "models-xinyue/pythia-70m-step140000-to-pythia-410m" \
#     --lora_path "models-xinyue/pythia-70m-step140000-to-pythia-410m-1b-lora-1e5" \
#     --tokenizer_path "EleutherAI/pythia-70m" \
#     --eval_results_path "eval/r64_70m_step140000_410m_eval_results_1b" \
#     --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \


# python src/eval_llm.py \
#     --base_model_path "EleutherAI/pythia-410m" \
#     --tokenizer_path "EleutherAI/pythia-410m" \
#     --eval_results_path "eval/410m_eval_results" \
#     --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \



# --lora_path "models/pythia-70m-to-pythia-410m-lora"
# --base_model_path "EleutherAI/pythia-70m" \
# --tokenizer_path "EleutherAI/pythia-70m" \


# --base_model_path "models/pythia-70m-to-pythia-410m" \
    # --lora_path "models/pythia-70m-to-pythia-410m-lora" \