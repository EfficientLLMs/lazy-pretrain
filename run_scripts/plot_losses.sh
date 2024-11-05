#!/bin/bash



python src/visualize/plot_losses.py \
    --log ".slurm_logs/1b_full_70m_410m_step140000-1e-3.out" \
    --plot_title "Pretraining loss for 1b_full_70m_410m_step140000-1e-3" \
    --output "plots/1b_full_70m_410m_step140000-1e-3.png"