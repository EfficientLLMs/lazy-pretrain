#!/bin/bash
#SBATCH --job-name=eval_pythia-410m_all
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=.slurm_logs/eval_pythia-410m_all.out
#SBATCH --time=02-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuel4@andrew.cmu.edu

# First, create exponential steps: 0,1,2,4,8,16,32,64,128,256,512
for i in {0..9}; do
  exp_steps="$exp_steps $(( 2**i ))"
done

# Then add 1000 and subsequent steps every 1000 up to 143000
for ((i=1000; i<=143000; i+=1000)); do
  linear_steps="$linear_steps $i"
done

# Combine and sort all steps
all_steps=$(echo "0 $exp_steps $linear_steps" | tr ' ' '\n' | sort -n | uniq)

# Evaluate each checkpoint
for step in $all_steps; do

  # Skip if results file already exists
  if [ -f "eval_410/410m_step${step}_eval_results.json" ]; then
      echo "Skipping step${step} - results file already exists"
      continue
  fi

  echo "Evaluating checkpoint: step${step}"
  
  python src/eval_llm.py \
    --base_model_path "EleutherAI/pythia-410m" \
    --checkpoint_step "step${step}" \
    --tokenizer_path "EleutherAI/pythia-410m" \
    --eval_results_path "eval_410/410m_step${step}_eval_results" \
    --tasks "lambada_openai" \
    --token ".token"

  # clean up
  # rm -rf ~/.cache/huggingface/hub/models--EleutherAI--pythia-410m
  # rm -rf ~/.cache/huggingface/hub/models--EleutherAI--pythia-70m
done