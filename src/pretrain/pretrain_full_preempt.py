import argparse
from datasets import load_dataset
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb
import os
import logging
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

# Relative imports
from utils import (
    CustomBinFileDataset, 
    seed_all, 
    cleanup_memory, 
    ResumableSampler, 
    CheckpointManager, 
    count_parameters,
    train_full_preempt
)



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for pretraining')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for pretraining')
    
    parser.add_argument('--output_dir', type=str, default='models/pythia-70m-to-pythia-410m',
                        help='Directory to save the trained model')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/pythia-70m-to-pythia-410m-relora',
                    help='Directory to save the trained model')

    parser.add_argument('--debug', action='store_true',
                    help='Debug mode')  


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

    # wandb
    parser.add_argument('--wandb_entity', type=str, default='vibhamasti',
                        help='Entity for wandb logging')
    parser.add_argument('--wandb_run_name', type=str, default='relora-8b',)


    # checkpoint
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                      help='How often to save checkpoints (in steps)')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    
    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")

    # Initialize checkpoint manager and try to load checkpoint
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, keep_last_n=2)
    checkpoint = checkpoint_manager.load_latest_checkpoint()

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
    sampler = ResumableSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        # shuffle=False,  # keep the same order
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=1,
        pin_memory=True
    )

    total_batches = len(dataloader)
    num_gpus = accelerator.num_processes
    steps_per_gpu = total_batches // num_gpus

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.grown_model, device_map=device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    
    # print trainable parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Train ratio: {trainable_params / total_params:.4f}")

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}, shape = {param.shape}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=args.lr
    )


    # Load checkpoint
    if checkpoint is not None:
        logger.info("Loading checkpoint states...")
        cleanup_memory()

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.AdamW(
            [p for n, p in unwrapped_model.named_parameters() if p.requires_grad],
            lr=args.lr
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['sampler_state']:
            sampler.load_state_dict(checkpoint['sampler_state'])

        cleanup_memory()
        logger.info(f"Resuming from step {start_step}")
    else:
        start_step = 0
        logger.info("Starting training from beginning")

    # Prepare for accelerator
    print("\nPreparing for accelerator...")
    model = model.to(device)
    cleanup_memory()

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )


    # wandb
    if accelerator.is_main_process:
        print(f"\nTraining configuration:")
        print(f"Total batches: {total_batches}")
        print(f"Number of GPUs: {num_gpus}")
        print(f"Steps per GPU: {steps_per_gpu}")
        print(f"Total steps: {steps_per_gpu * num_gpus}")


        wandb.init(
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            project="full-pretraining-preempt",
            config={
                "model": args.grown_model,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_tokens": args.num_tokens,
                "chunk_size": args.chunk_size,
                "total_steps_per_gpu": steps_per_gpu,
            }
        )

    
    # Train model
    train_full_preempt(model, accelerator, dataloader, optimizer, args, 
                 checkpoint_manager=checkpoint_manager, start_checkpoint=checkpoint, sampler=sampler)
    

if __name__ == '__main__':
    main()