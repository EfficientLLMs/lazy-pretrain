import argparse
from datasets import load_dataset
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import Accelerator
import wandb
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

# Relative imports
from utils import CustomBinFileDataset, seed_all, train
from relora import ReLoRaModel
from relora_utils import get_scheculer, train_relora


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

    parser.add_argument('--rank', type=int, default=256,
                        help='Rank of the Lora module')
    
    parser.add_argument('--lora_alpha', type=int, default=256,
                        help='Alpha of the lora module (\\deltaW * \\alpha/r)')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for pretraining')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for pretraining')
    
    parser.add_argument('--output_dir', type=str, default='models/pythia-70m-to-pythia-410m-lora',
                        help='Directory to save the trained model')

    # on-the-fly data processing
    parser.add_argument('--use_on_the_fly', action='store_true',
                        help='Whether to use on-the-fly data processing')
    parser.add_argument('--first_idx', type=int, default=19, 
                        help='The index of the first .bin file')
    parser.add_argument('--last_idx', type=int, default=20, 
                        help='The index of the last .bin file')
    parser.add_argument('--num_tokens', type=int, default=int(1e9),
                        help='The target number of tokens of the dataset')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Chunk size for tokenized content')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode')  

    # relora
    parser.add_argument('--scheduler', type=str, default='cosine_restarts')
    parser.add_argument('--relora_steps', type=int, default=1500)
    parser.add_argument('--cycle_length', type=int, default=1500)
    parser.add_argument('--warmup_steps', type=int, default=150)
    parser.add_argument('--restart_warmup_steps', type=int, default=150)  # same as the intial warmup
    parser.add_argument('--min_lr_ratio', type=float, default=0.1)  # decay to 0.1 * lr
    
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.grown_model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")

    # wandb
    if accelerator.is_main_process:
        wandb.init(
            project="relora-pretraining",
            config={
                "model": args.grown_model,
                "rank": args.rank,
                "lora_alpha": args.lora_alpha,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "scheduler": args.scheduler,
                "relora_steps": args.relora_steps,
                "cycle_length": args.cycle_length,
                "warmup_steps": args.warmup_steps,
                "min_lr_ratio": args.min_lr_ratio,
            }
        )

    # Use ReLoRA
    model = ReLoRaModel(
        model,
        target_modules="all-linear",
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        # it doesn't have these parameters
        # bias="none",  
        # task_type="CAUSAL_LM",
        keep_original_weights=True,
        lora_only=False,
        trainable_scaling=False,
    )

    # Load dataset
    if args.use_on_the_fly:
        dataset = CustomBinFileDataset(
            first_idx=args.first_idx,
            last_idx=args.last_idx,
            num_tokens=args.num_tokens,
            chunk_size=args.chunk_size,
            debug=args.debug
        )
    else:
        dataset = torch.load(args.dataset)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        shuffle=False,  # keep the same order
        collate_fn=default_data_collator,
        num_workers=1,
        pin_memory=True
    )
    total_steps = len(dataloader)
    if accelerator.is_main_process:
        print(f"Total training steps: {total_steps}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = get_scheculer(
        optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=total_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        cycle_length=args.cycle_length,
        restart_warmup_steps=args.restart_warmup_steps
    )

    # Prepare for accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, 
        optimizer, 
        dataloader, 
        scheduler
    )

    # Train model
    train_relora(model, accelerator, dataloader, optimizer, scheduler, args, args.output_dir)

    

if __name__ == '__main__':
    main()