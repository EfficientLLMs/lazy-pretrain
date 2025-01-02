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
import os
import signal
import threading
from subprocess import call

# Relative imports
from utils import CustomBinFileDataset, seed_all, train
from relora import ReLoRaModel
from relora_utils import get_scheculer, train_relora, optimizer_reset


# Add RequeueModelSaver for preemption handling
# ^ This class stores training state, and use a lock to ensure that only one process saves the model
class RequeueModelSaver:
    def __init__(self):
        self.model = None
        self.save_lock = threading.Lock()
        self.has_saved = False
        self.current_step = 0

    def set_model(self, model, accelerator, optimizer, scheduler, args):
        self.model = model
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def update_step(self, step):
        self.current_step = step

    def __call__(self):
        with self.save_lock:
            if self.has_saved:
                return
            self.has_saved = True
            
            # Custom save function
            if self.accelerator.is_main_process:
                checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint-{self.current_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # ^ This saves the model
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                    checkpoint_path,
                )
                
                # ^ This saves the training state
                self.accelerator.save({
                    'step': self.current_step,
                    'optimizer': self.accelerator.get_state_dict(self.optimizer),
                    'scheduler': self.scheduler.state_dict(),
                }, os.path.join(checkpoint_path, "training_state.pt"))
                
                print(f"Saved checkpoint at step {self.current_step}")


# Initialize global model saver and save flag
save_model = RequeueModelSaver()
save_after_batch = False


# ^ Set a flag when a preemption signal is received
def slurm_requeue_handler(signum, frame):
    global save_after_batch
    print(f"Received signal {signum}, will save after current batch completes...")
    save_after_batch = True


def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, latest)


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
    args = parser.parse_args()
    return args


def train_relora_preempt(model, accelerator, dataloader, optimizer, scheduler, args, output_dir):
    """
    Training loop for ReLoRA with preemption handling.
    """
    global save_after_batch

    # Set up SLURM handlers if running on SLURM
    if os.environ.get("SLURM_JOB_ID"):
        print("Running in SLURM, setting up requeue handler...")
        signal.signal(signal.SIGUSR1, slurm_requeue_handler)
        signal.signal(signal.SIGTERM, slurm_requeue_handler)

    # Initialize model saver
    save_model.set_model(model, accelerator, optimizer, scheduler, args)

    model.train()

    # Debug prints from all processes
    print(f"\nProcess {accelerator.process_index} debug info:")
    print(f"  Device: {accelerator.device}")
    print(f"  Is main process: {accelerator.is_main_process}")
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Is model training: {model.training}")

    # Synchronize before starting training loop
    print(f"\nProcess {accelerator.process_index} waiting for synchronization...")
    accelerator.wait_for_everyone()
    print(f"Process {accelerator.process_index} synchronized")

    if accelerator.is_main_process:
        print("\nStarting training loop...")
        progress_bar = tqdm(total=len(dataloader), desc="Training")
    
    # ^ The following is the same until we receive a preemption signal
    for step, batch in enumerate(dataloader):
        try:
            # ReLoRA reset
            if step > 0 and step % args.relora_steps == 0:
                if accelerator.is_main_process:
                    print(f"\nPerforming LoRA reset at step {step}")
                
                # Synchronize before model manipulation
                accelerator.wait_for_everyone()
                
                # 1. Get unwrapped model and perform reset
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.merge_and_reinit()
                
                # 2. Reset optimizer state for LoRA parameters
                lora_params = [p for n, p in unwrapped_model.named_parameters() if "lora_" in n]
                optimizer_reset(
                    optimizer,
                    reset_params=lora_params,
                    optimizer_state_keys=["exp_avg", "exp_avg_sq"],
                    reset_optimizer_on_relora=True,
                    optimizer_random_pruning=0.0,
                    optimizer_magnitude_pruning=0.0,
                )
                
                # Synchronize after model manipulation
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    wandb.log({"relora_reset": step}, step=step)
                    print("LoRA reset complete")

            # Regular training step
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
            loss = outputs.loss
            
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Get loss values
            loss_value = accelerator.gather(loss).mean().item()
            ppl_value = torch.exp(torch.tensor(loss_value)).item()
            current_lr = scheduler.get_last_lr()[0]

            # ^ Update save_model step counter
            if accelerator.is_main_process:
                print(f"Update step={step} for save_model")
            save_model.update_step(step)

            # Logging
            if accelerator.is_main_process:
                wandb.log({
                    "loss": loss_value,
                    "ppl": ppl_value,
                    "learning_rate": current_lr,
                    "step": step,
                }, step=step)
                
                progress_bar.update(1)
                if step % 100 == 0:
                    print(f"Step {step}, loss: {loss_value:.4f}, ppl: {ppl_value:.2f}, lr: {current_lr:.2e}")

            # Check if we need to save and requeue
            # ^ This is True if we received a preemption signal
            if save_after_batch:
                if accelerator.is_main_process:
                    print("\nReceived preemption signal, saving checkpoint and requeuing...")
                
                # Synchronize before saving
                accelerator.wait_for_everyone()
                
                # Save checkpoint
                save_model()
                
                # ^ After each batch processing, we first check if we received a preemption signal
                # ^ If we did, we save the model and manually requeue the job
                if os.environ.get("SLURM_JOB_ID"):
                    job_id = os.environ["SLURM_JOB_ID"]
                    if accelerator.is_main_process:
                        print(f"Requeuing job {job_id}")
                        try:
                            call(["scontrol", "requeue", job_id])
                        except FileNotFoundError:
                            call(f"scontrol requeue {job_id}", shell=True)
                
                # # Close progress bar and wandb if main process
                # if accelerator.is_main_process:
                #     progress_bar.close()
                #     wandb.finish()
                
                return None  # Exit training loop

            if args.debug and step > 5:
                break

        except Exception as e:
            print(f"Error during training step {step}: {str(e)}")
            raise e

    # Synchronize before finishing
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Training finished.")
        progress_bar.close()
        wandb.finish()

        # Save model
        if not args.debug:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )

    return None  # Return values aren't needed in distributed setting


def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Accelerator
    accelerator = Accelerator()
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

        wandb_run = wandb.init(
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
            },
            # ^ Allow resuming training
            id=os.environ.get("WANDB_RUN_ID", None),
            resume="allow",
        )
        # save the id
        os.environ["WANDB_RUN_ID"] = wandb_run.id


    # Check for latest checkpoint
    latest_checkpoint = find_latest_checkpoint(args.output_dir)
    starting_step = 0

    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(latest_checkpoint, device_map=device)
        
        training_state = torch.load(
            os.path.join(latest_checkpoint, "training_state.pt"),
            map_location='cpu'
        )
        starting_step = training_state['step']

        # Skip processed batches in dataloader
        dataloader = DataLoader(
            torch.utils.data.Subset(
                dataloader.dataset,
                range(starting_step * args.batch_size, len(dataloader.dataset))
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
            num_workers=1,
            pin_memory=True
        )
    else:
        print("Starting fresh training")
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=args.lr
    )

    # Scheduler
    scheduler = get_scheculer(
        optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=steps_per_gpu * num_gpus,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        cycle_length=cycle_length,
        restart_warmup_steps=warmup_steps // 2,
    )

    # Load optimizer and scheduler states if resuming
    if latest_checkpoint and training_state:
        optimizer.load_state_dict(training_state['optimizer'])
        if training_state.get('scheduler'):
            scheduler.load_state_dict(training_state['scheduler'])

    # Prepare for accelerator
    print("\nPreparing for accelerator...")
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Train model
    train_relora_preempt(
        model, 
        accelerator, 
        dataloader, 
        optimizer, 
        scheduler, 
        args,
        args.output_dir
    )



if __name__ == '__main__':
    main()