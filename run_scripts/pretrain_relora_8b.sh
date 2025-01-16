#!/bin/bash
#SBATCH --job-name=pythia-70m-step139000-8b-relora
#SBATCH --mem=80G
#SBATCH --gres=gpu:A6000:8
#SBATCH --partition=preempt
#SBATCH --output=.slurm_logs/pythia-70m-step139000-8b-relora.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu
#SBATCH --exclude=shire-1-6,shire-1-10,babel-1-27,babel-1-23


# export NCCL_P2P_LEVEL=NVL

export NCCL_P2P_DISABLE=1


# Define variables
EXP_NAME="8b_relora-1e-5"
STEP="step139000"
MODEL_NAME="pythia-70m-"$STEP"-to-pythia-410m"

GROWN_MODEL="models/"$MODEL_NAME
TRAINED_MODEL="models/"$MODEL_NAME"-"$EXP_NAME
CHECKPOINT_DIR="checkpoints/"$MODEL_NAME"-"$EXP_NAME

accelerate launch src/pretrain/pretrain_relora_preempt.py \
    --grown_model $GROWN_MODEL \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --rank 256 \
    --lora_alpha 256 \
    --batch_size 16 \
    --lr 1e-5 \
    --output_dir $TRAINED_MODEL \
    --checkpoint_dir $CHECKPOINT_DIR \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 8_000_000_000 \
    --chunk_size 512 \
    --scheduler "cosine_restarts" \
    --min_lr_ratio 0.1 \
    --wandb_entity "vibhamasti" \
    --num_restarts 40 \
    --checkpoint_freq 10 \
    --wandb_run_name "step139000-8b-relora"
    # --do_extact_lora

    # we don't need others; but we need to specify the num_restarts in the main code
    # --relora_steps 350 \
    # --cycle_length 350 \
    # --warmup_steps 50 \
    # --restart_warmup_steps 50 \


echo "Evaluating "$TRAINED_MODEL

python src/eval_llm.py \
    --base_model_path $TRAINED_MODEL \
    --tokenizer_path "EleutherAI/pythia-70m" \
    --eval_results_path "eval/eval_"$EXP_NAME"-"$MODEL_NAME \
    --tasks "paloma" "lambada_openai" \
    --token ".token"

echo "Finished evaluating full model. Evaluation results saved in eval/eval_"$EXP_NAME"-"$MODEL_NAME
