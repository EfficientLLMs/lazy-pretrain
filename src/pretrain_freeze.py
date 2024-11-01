import argparse
from datasets import load_dataset
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import Accelerator
import wandb
import os
import gc
import json
from tqdm import tqdm
import pickle
from torch.nn.utils import clip_grad_norm_
from eval_llm import evaluate_model

# Relative imports
from utils import CustomBinFileDataset, seed_all, train


def parse_args():

    parser = argparse.ArgumentParser(description='Continue pretraining a grown model')

    parser.add_argument('--grown_model', type=str, default='models/pythia-70m-to-pythia-410m',
                        help='Path to the model to continue pretraining')

    parser.add_argument('--tokenizer', type=str, default='EleutherAI/pythia-70m',
                        help='Tokenizer to use for the model')
    
    parser.add_argument('--dataset', type=str, default='data/train_00/00_5m.pt',
                        help='Directory/hf name containing the pretraining data')
    
    parser.add_argument('--seed', type=int, default=1234,
                        help='Seed for sampling the dataset')

    # parser.add_argument( "--tasks", type=str, nargs="+", 
    #                     required=True, default=["lambada_openai", "arc_easy"])

    # parser.add_argument("--num_fewshot", type=int, default=0)

    # parser.add_argument("--eval_results_path", type=str, 
    #                     default="eval/evaluation_results")

    # parser.add_argument("--parallelize", action="store_true")

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for pretraining and eval')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for pretraining')
    
    parser.add_argument('--output_dir', type=str, default='models/pythia-70m-to-pythia-410m-lora',
                        help='Directory to save the trained model')
    
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.grown_model)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Copied layers pickle
    copied_layers_file = os.path.join(args.grown_model, "copied_layers.pkl")
    with open(copied_layers_file, "rb") as f:
        copied_layers = pickle.load(f)

    # Accelerator
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    print(f"device: {device}")

    # Freeze appropriate layers
    for i in range(len(copied_layers)):
        if not copied_layers[i]:
            parent_module = model.gpt_neox.layers[i]

            # Freeze the weights of the parent module
            for name, param in parent_module.named_parameters():
                param.requires_grad = False
        else:
            parent_module = model.gpt_neox.layers[i]

            # Freeze the weights of the parent module
            for name, param in parent_module.named_parameters():
                param.requires_grad = True
            

    # Enable gradient checkpointing (for memory efficiency)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Load dataset
    dataset = torch.load(args.dataset)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=default_data_collator,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Prepare for accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Train model
    train(model, accelerator, dataloader, optimizer, args.output_dir)

    # Delete the model to free up memory
    # First move the model to CPU
    model = model.to('cpu')
    del model

    # Clear up GPU and memory
    torch.cuda.empty_cache()
    gc.collect()
    

    

if __name__ == '__main__':
    main()