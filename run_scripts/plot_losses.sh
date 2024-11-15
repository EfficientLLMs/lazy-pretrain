#!/bin/bash

# Define variables
EXP_NAME="1b_full-1e-5"
STEP="step143000"
MODEL_NAME="pythia-70m-"$STEP"-to-pythia-410m"

GROWN_MODEL="models/"$MODEL_NAME
TRAINED_MODEL="models/"$MODEL_NAME"-"$EXP_NAME

python src/visualize/plot_losses.py \
    --log ".slurm_logs/"$EXP_NAME"-"$MODEL_NAME".out" \
    --plot_title "Pretraining loss for "$EXP_NAME"-"$MODEL_NAME \
    --output "plots/"$EXP_NAME"-"$MODEL_NAME".png" \
