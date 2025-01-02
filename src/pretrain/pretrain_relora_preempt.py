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
import sys
import logging
from pathlib import Path
import json
import os

# Relative imports
from utils import CustomBinFileDataset, seed_all, train
from relora import ReLoRaModel
from relora_utils import get_scheculer, train_relora_preempt


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Data sampler
class ResumableSampler(torch.utils.data.Sampler):
    """Sampler that supports resuming from a specific index"""
    def __init__(self, data_source, start_idx=0):
        self.data_source = data_source
        self.start_idx = start_idx
        
    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))
        
    def __len__(self):
        return len(self.data_source) - self.start_idx
        
    def state_dict(self):
        return {'start_idx': self.start_idx}
        
    def load_state_dict(self, state_dict):
        self.start_idx = state_dict['start_idx']


# Manage checkpoint
class CheckpointManager:
    """Simple checkpoint manager to handle saving and loading of checkpoints"""
    def __init__(self, output_dir, keep_last_n=2):
        self.output_dir = Path(output_dir)
        self.keep_last_n = keep_last_n
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, state, step):
        checkpoint_path = self.output_dir / f"checkpoint_{step}"
        tmp_checkpoint_path = checkpoint_path.with_suffix('.tmp')
        
        # First save to a temporary file
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state, tmp_checkpoint_path)
        
        # Atomic rename
        tmp_checkpoint_path.rename(checkpoint_path)
        
        # Save latest checkpoint info
        with open(self.output_dir / "latest_checkpoint.json", 'w') as f:
            json.dump({
                'step': step,
                'checkpoint_path': str(checkpoint_path)
            }, f)
            
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def load_latest_checkpoint(self):
        latest_file = self.output_dir / "latest_checkpoint.json"
        if not latest_file.exists():
            logger.info("No checkpoint found - starting from beginning")
            return None
            
        with open(latest_file, 'r') as f:
            latest_info = json.load(f)
            
        checkpoint_path = Path(latest_info['checkpoint_path'])
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file {checkpoint_path} not found")
            return None
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        return torch.load(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove all but the latest n checkpoints"""
        checkpoints = sorted(
            [f for f in self.output_dir.glob("checkpoint_*") if not f.name.endswith('.tmp')],
            key=lambda x: int(x.name.split('_')[1])
        )
        
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                logger.info(f"Removing old checkpoint: {checkpoint}")
                checkpoint.unlink()


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
    parser.add_argument('--num_restarts', type=int, default=5)
    parser.add_argument('--do_extact_lora', action='store_true',
                        help='Whether to train only the lora_A and lora_B')

    # wandb
    parser.add_argument('--wandb_entity', type=str, default='irisiris',
                        help='Entity for wandb logging')

    # checkpoint
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                      help='How often to save checkpoints (in steps)')

    args = parser.parse_args()
    return args

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")

    # Initialize checkpoint manager and try to load checkpoint
    checkpoint_manager = CheckpointManager(args.output_dir, keep_last_n=2)
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

    # This should be specified based on the total number of tokens
    # For 1B, I use 5 restarts
    num_restarts = args.num_restarts
    desired_cycle_length = steps_per_gpu // num_restarts
    cycle_length = desired_cycle_length * num_gpus
    
    desired_warmup_per_gpu = desired_cycle_length // num_restarts
    warmup_steps = desired_warmup_per_gpu * num_gpus

    # update the args
    args.relora_steps = cycle_length // num_gpus  # yeah this is confusing
    args.warmup_steps = warmup_steps
    args.restart_warmup_steps = warmup_steps // 2
    args.cycle_length = cycle_length

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.grown_model, device_map=device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Use ReLoRA
    model = ReLoRaModel(
        model,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
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

    # Note: the default relora config set everything to be trainable, except it sets the dense weight to be lora_A and lora_B
    if args.do_extact_lora:
        # Set all parameters to be non-trainable
        # then set only the lora_A and lora_B 
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True

    # print trainable parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Train ratio: {trainable_params / total_params:.4f}")

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}, shape = {param.shape}")

    # sys.exit()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=args.lr
    )

    # Scheduler
    if checkpoint is not None:
        start_step = checkpoint['global_step'] + 1
        effective_step = start_step - checkpoint['last_relora_reset']
        
        scheduler = get_scheculer(
            optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=steps_per_gpu * num_gpus,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            cycle_length=cycle_length,
            restart_warmup_steps=warmup_steps // 2,
            adjust_step=effective_step
        )
    else:
        scheduler = get_scheculer(
            optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=steps_per_gpu * num_gpus,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            cycle_length=cycle_length,
            restart_warmup_steps=warmup_steps // 2,
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
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # wandb
    if accelerator.is_main_process:
        print(f"\nTraining configuration:")
        print(f"Total batches: {total_batches}")
        print(f"Number of GPUs: {num_gpus}")
        print(f"Steps per GPU: {steps_per_gpu}")
        print(f"Total steps: {steps_per_gpu * num_gpus}")
        print(f"Number of restarts: {num_restarts}")
        print(f"Cycle length: {cycle_length}")
        print(f"Warmup steps: {warmup_steps}")

        wandb.init(
            entity=args.wandb_entity,
            project="relora-pretraining-preempt",
            config={
                "model": args.grown_model,
                "rank": args.rank,
                "lora_alpha": args.lora_alpha,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "scheduler": args.scheduler,
                "relora_steps": cycle_length,
                "cycle_length": cycle_length,
                "warmup_steps": warmup_steps,
                "min_lr_ratio": args.min_lr_ratio,
                "total_steps_per_gpu": steps_per_gpu,
                "num_restarts": num_restarts,
                "resumed_from_step": start_step if checkpoint else 0
            }
        )

    # Train model
    train_relora_preempt(model, accelerator, dataloader, optimizer, scheduler, args, 
                 checkpoint_manager=checkpoint_manager, start_checkpoint=checkpoint, sampler=sampler)

    

if __name__ == '__main__':
    main()