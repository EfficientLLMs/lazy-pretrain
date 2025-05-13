import argparse
from datasets import load_dataset
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs
import wandb
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import sys

# Relative imports
from utils import CustomBinFileDataset, seed_all, train, CustomDolmaDataset, ResumableSampler
from prepare_memmap import map_number_to_text


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


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
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')

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
    # wandb
    parser.add_argument('--wandb_entity', type=str, default='vibhamasti',
                        help='Entity for wandb logging')
    parser.add_argument('--wandb_run_name', type=str, default='lazy-pretraining',
                        help='Run name for wandb')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1,
                           mixed_precision="fp16",
                           kwargs_handlers=[GradScalerKwargs(init_scale=2**14, enabled=True)])
    device = accelerator.device
    print(f"device: {device}")

    # Load dataset
    if args.use_on_the_fly:
        dataset = CustomBinFileDataset(
            first_idx=args.first_idx,
            last_idx=args.last_idx,
            num_tokens=args.num_tokens,
            chunk_size=args.chunk_size,
            debug=args.debug
        )
        # total_tokens = len(dataset) * args.chunk_size
        # assert total_tokens == args.num_tokens
    elif args.dataset == 'dolma':
        dataset = CustomDolmaDataset(
            memmap_file=f"data/dolma_tokenized/{map_number_to_text(args.num_tokens)}.npy", 
            chunk_size=args.chunk_size, 
            debug=False, 
            num_tokens=args.num_tokens
        )
    else:
        dataset = torch.load(args.dataset)

    # sample = dataset[0]
    # print(f"  Shape: {sample['input_ids'].shape}")
    # print(f"  Data type: {sample['input_ids'].dtype}")
    # print(f"  First few tokens: {sample['input_ids'][:10]}")

    # Create dataloader
    sampler = ResumableSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        shuffle=False,  # keep the same order
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=0,
        pin_memory=True
    )

    total_batches = len(dataloader)
    num_gpus = accelerator.num_processes
    steps_per_gpu = total_batches // num_gpus

    # wandb
    if accelerator.is_main_process:
        print(f"\nTraining configuration:")
        print(f"Total batches: {total_batches}")
        print(f"Number of GPUs: {num_gpus}")
        print(f"Steps per GPU: {steps_per_gpu}")
        print(f"Total steps: {steps_per_gpu * num_gpus}")

        wandb.init(
            entity=args.wandb_entity,
            project="lora-pretraining",
            name=args.wandb_run_name,
            config={
                "model": args.grown_model,
                "rank": args.rank,
                "lora_alpha": args.lora_alpha,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "total_steps_per_gpu": steps_per_gpu,
            }
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.grown_model,
        device_map=device,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    

    if torch.cuda.is_available():
        print(f"Memory after enabling gradient checkpointing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Model memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Add lora module to model
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["query_key_value"],
        # target_modules=["att_proj"],
        # target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Add lora module to model
    model = get_peft_model(model, config)

    total_params, trainable_params = count_parameters(model)
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Train ratio: {trainable_params / total_params:.4f}")

    for name, param in model.named_parameters():
        if param.requires_grad: 
            param.data = param.data.float()
        print(f"{name}: requires_grad = {param.requires_grad}, shape = {param.shape}, type = {param.dtype}")

    # sys.exit()

    

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=args.lr
    )

    # Prepare for accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Train model
    train(
        model, 
        accelerator, 
        dataloader, 
        optimizer, 
        args.output_dir, 
        debug=args.debug, 
        autocast=True,
        checkpoint_dir=args.checkpoint_dir
    )

    

if __name__ == '__main__':
    main()
