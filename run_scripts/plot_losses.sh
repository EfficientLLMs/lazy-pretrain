#!/bin/bash



python src/visualize/plot_losses.py \
    --log ".slurm_logs/1b_r256_70m_410m_lora.out" \
    --plot_title "Pretraining loss for 1b_r256_70m_410m_lora_1e-5" \
    --output "plots/1b_r256_70m_410m_lora_1e-5.png"