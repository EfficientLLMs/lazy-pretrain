#!/bin/bash
#SBATCH --job-name=olmo-1b-7b-relora-5b-tokens
#SBATCH --mem=80G
#SBATCH --gres=gpu:L40S:8
#SBATCH --partition=general
#SBATCH --output=.slurm_logs/olmo-1b-7b-relora-5b-tokens.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuel4@andrew.cmu.edu

export NCCL_P2P_DISABLE=1


# Define variables
EXP_NAME="relora_5b_1e-5"
MODEL_NAME="OLMo-1B-to-OLMo-7B"

GROWN_MODEL="models/"$MODEL_NAME
TRAINED_MODEL="models/"$MODEL_NAME"_"$EXP_NAME

# NOTE: this won't work cause by default relora is setting everything to be trainable
accelerate launch src/pretrain/pretrain_relora.py \
    --grown_model $GROWN_MODEL \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 1 \
    --lr 1e-5 \
    --output_dir $TRAINED_MODEL \
    --dataset "dolma" \
    --num_tokens 5_000_000_000 \
    --chunk_size 512 \
    --scheduler "cosine_restarts" \
    --min_lr_ratio 0.1 \
    --wandb_entity "irisiris" \
    --num_restarts 25 \
    --do_extact_lora

    # we don't need others; but we need to specify the num_restarts in the main code
    # --relora_steps 350 \
    # --cycle_length 350 \
    # --warmup_steps 50 \
    # --restart_warmup_steps 50 \


# echo "Evaluating "$TRAINED_MODEL

# python src/eval_llm.py \
#     --base_model_path $TRAINED_MODEL \
#     --tokenizer_path "EleutherAI/pythia-70m" \
#     --eval_results_path "eval/eval_"$EXP_NAME"-"$MODEL_NAME \
#     --tasks "paloma" "lambada_openai" \
#     --token ".token"

# echo "Finished evaluating full model. Evaluation results saved in eval/eval_"$EXP_NAME"-"$MODEL_NAME
