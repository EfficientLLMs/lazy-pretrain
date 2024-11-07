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
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

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

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for pretraining')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for pretraining')
    
    parser.add_argument('--output_dir', type=str, default='models/pythia-70m-to-pythia-410m',
                        help='Directory to save the trained model')

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
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume training from a checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/pythia-70m-to-pythia-410m',
                        help='Directory to save the checkpoints')
    
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Load model
    if args.checkpoint_dir is not None:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

    if args.resume:
        # Find the latest checkpoint
        checkpoints = os.listdir(args.checkpoint_dir)
        tokens_seen = [int(ckpt.split(".")[0]) for ckpt in checkpoints]
        latest_checkpoint = max(tokens_seen)
        
        # Load the latest checkpoint
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir + f"/{latest_checkpoint}.pt")
        print(f"Resuming training from checkpoint {latest_checkpoint}")

        args.num_tokens = args.num_tokens - latest_checkpoint

    else:
        model = AutoModelForCausalLM.from_pretrained(args.grown_model)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Accelerator
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    print(f"device: {device}")


    # Enable gradient checkpointing
    print("Enabling gradient checkpointing")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Load dataset
    if args.use_on_the_fly:
        dataset = CustomBinFileDataset(
            first_idx=args.first_idx,
            last_idx=args.last_idx,
            num_tokens=args.num_tokens,
            chunk_size=args.chunk_size
        )
        total_tokens = len(dataset) * args.chunk_size
        assert total_tokens == args.num_tokens
    else:
        dataset = torch.load(args.dataset)


    print(f"Dataset size: {len(dataset)}")
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        shuffle=False,  # keep the same order
        collate_fn=default_data_collator,
        num_workers=1,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("Model, optimizer, dataloader created")

    # Prepare for accelerator
    print("Preparing for accelerator")
    # model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    print("Preparing model")
    model = accelerator.prepare(model)

    print("Preparing optimizer")
    optimizer = accelerator.prepare(optimizer)

    print("Preparing dataloader")
    dataloader = accelerator.prepare(dataloader)

    # Train model
    print("Training model")
    train(model, accelerator, dataloader, optimizer, args.output_dir, debug=False, checkpoint_dir=args.checkpoint_dir)
    

if __name__ == '__main__':
    main()