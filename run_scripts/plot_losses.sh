#!/bin/bash



python src/visualize/plot_losses.py \
    --log ".slurm_logs/pythia-70m-step140000-1b-lora-alpha256-allmod-1e-5.out" \
    --plot_title "Pretraining loss for pythia-70m-step140000-1b-lora-alpha256-allmod-1e-5.out" \
    --output "plots/pythia-70m-step140000-1b-lora-alpha256-allmod-1e-5.png"