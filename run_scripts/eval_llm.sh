#!/bin/bash
#SBATCH --job-name=eval_freeze_70m_step140000_410m_tokens_1b_alpha256-allmod
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:2
#SBATCH --output=.slurm_logs/eval_freeze_70m_step140000_410m_tokens_1b_alpha256-allmod.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu



python src/eval_llm.py \
    --base_model_path "models/pythia-70m-step140000-to-pythia-410m" \
    --lora_path "models/pythia-70m-step140000-to-pythia-410m-lora-alpha256-allmod" \
    --tokenizer_path "EleutherAI/pythia-70m" \
    --eval_results_path "eval/lora_70m_step140000_410m_eval_results_1b_alpha256-allmod" \
    --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa"


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

python src/eval_llm.py \
    --base_model_path "EleutherAI/pythia-70m" \
    --checkpoint_step "step140000" \
    --tokenizer_path "EleutherAI/pythia-70m" \
    --eval_results_path "eval/70m_step140000_eval_results" \
    --tasks "lambada_openai" "arc_easy" "arc_challenge" "hellaswag" "piqa" "winogrande" "sciq" "logiqa" "logiqa2" "openbookqa" \


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