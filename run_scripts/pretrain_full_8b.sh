#!/bin/bash
#SBATCH --job-name=pythia-70m-step139000-8b-full
#SBATCH --mem=80G
#SBATCH --gres=gpu:A6000:8
#SBATCH --partition=preempt
#SBATCH --output=.slurm_logs/pythia-70m-step139000-8b-full.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --exclude=shire-1-6,shire-1-10,babel-1-27
#SBATCH --mail-user=vmasti@andrew.cmu.edu


# Define variables
EXP_NAME="8b_full-1e-5"
STEP="step139000"
MODEL_NAME="pythia-70m-"$STEP"-to-pythia-410m"

GROWN_MODEL="models/"$MODEL_NAME
TRAINED_MODEL="models/"$MODEL_NAME"-"$EXP_NAME
CHECKPOINT_DIR="checkpoints/"$MODEL_NAME"-"$EXP_NAME

# Grow model if grown model does not exist yet
if [ ! -d $GROWN_MODEL ]; then
    echo "Growing model..."
    python src/grow/grow.py \
        --small_model "pythia-70m" \
        --large_depth 24 \
        --large_width 1024 \
        --depth_growth "alternate" \
        --output_dir $GROWN_MODEL \
        --checkpoint_step $STEP
fi

echo "Grown model path: "$GROWN_MODEL

echo "Starting full model pretraining..."
# Pretrain full model
accelerate launch src/pretrain/pretrain_full_preempt.py \
    --grown_model $GROWN_MODEL \
    --tokenizer "EleutherAI/pythia-410m" \
    --seed 1234 \
    --batch_size 32 \
    --lr 1e-5 \
    --output_dir $TRAINED_MODEL \
    --checkpoint_dir $CHECKPOINT_DIR \
    --use_on_the_fly \
    --first_idx 19 \
    --last_idx 20 \
    --num_tokens 8_000_000_000 \
    --chunk_size 512 \
    --wandb_entity "vibhamasti" \
    --checkpoint_freq 10 \
    --wandb_run_name $STEP"-8b-full"
    
    # --checkpoint_dir "checkpoints/"$EXP_NAME"-"$MODEL_NAME

echo "Finished pretraining full model. Model saved in "$TRAINED_MODEL

echo "Evaluating "$TRAINED_MODEL

# Evaluate full model
python src/eval_llm.py \
    --base_model_path $TRAINED_MODEL \
    --tokenizer_path "EleutherAI/pythia-70m" \
    --eval_results_path "eval/eval_"$EXP_NAME"-"$MODEL_NAME \
    --tasks "paloma" "lambada_openai" \
    --token ".token"

echo "Finished evaluating full model. Evaluation results saved in eval/eval_"$EXP_NAME"-"$MODEL_NAME