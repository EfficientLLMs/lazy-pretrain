#!/bin/bash
#SBATCH --job-name=1b_grown_olmo_eval
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:8
#SBATCH --output=.slurm_logs/1b_grown_olmo_eval.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

# Define variables
# EXP_NAME="6b_full-1e-5"
# STEP="step140000"
# MODEL_NAME="pythia-70m-"$STEP"-to-pythia-410m"


python src/eval_llm.py \
    --base_model_path "models/OLMo-1B-to-OLMo-7B" \
    --tokenizer_path "allenai/OLMo-1B" \
    --eval_results_path "eval/OLMo-1B-to-OLMo-7B_eval_results" \
    --tasks "lambada_openai" "paloma_c4_100_domains" \
    --token ".token" \
    --batch_size 1

# python src/eval_llm.py \
#     --base_model_path "allenai/OLMo-1B" \
#     --tokenizer_path "allenai/OLMo-1B" \
#     --eval_results_path "eval/OLMo-1B_eval_results" \
#     --tasks "lambada_openai" "paloma_c4_100_domains" \
#     --token ".token" \
#     --batch_size 1
    # --checkpoint_path "checkpoints/pythia-70m-step139000-to-pythia-410m-8b_full-1e-5"

    # --checkpoint_step "step40000" \

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
