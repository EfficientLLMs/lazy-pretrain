import argparse
import torch
from transformers import AutoModelForCausalLM, default_data_collator
from accelerate import Accelerator
import wandb
from torch.utils.data import DataLoader
import logging

# Relative imports
from utils import CustomBinFileDataset, seed_all, cleanup_memory, ResumableSampler, CheckpointManager, count_parameters, CustomDolmaDataset
from relora import ReLoRaModel
from relora_utils import get_scheculer, train_relora_preempt
from prepare_memmap import map_number_to_text


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

    parser.add_argument('--rank', type=int, default=256,
                        help='Rank of the Lora module')
    
    parser.add_argument('--lora_alpha', type=int, default=256,
                        help='Alpha of the lora module (\\deltaW * \\alpha/r)')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for pretraining')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for pretraining')
    
    parser.add_argument('--output_dir', type=str, default='models/pythia-70m-to-pythia-410m-relora',
                        help='Directory to save the trained model')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/pythia-70m-to-pythia-410m-relora',
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

    # Initialize variables that will be loaded from checkpoint
    lora_state_dict = None
    optimizer_state = None
    scheduler_state = None
    sampler_state = None
    global_step = 0
    last_relora_reset = 0

    # Create checkpoint manager for all processes
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, keep_last_n=2)

    # First load checkpoint if it exists - only on main process
    if accelerator.is_main_process:
        checkpoint = checkpoint_manager.load_latest_checkpoint(device='cpu')  # Load to CPU first
        print("Main process loaded checkpoint")
        
        if checkpoint is not None:
            # Extract only LoRA parameters
            state_dict = checkpoint['model_state_dict']
            lora_state_dict = {k: v for k, v in state_dict.items() if 'lora_A' in k or 'lora_B' in k}
            
            # Extract other states
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            sampler_state = checkpoint.get('sampler_state', None)
            global_step = checkpoint['global_step']
            last_relora_reset = checkpoint['last_relora_reset']
            
            # Clear memory
            del state_dict
            cleanup_memory()
            torch.cuda.empty_cache()
    else:
        checkpoint = None
        lora_state_dict = None
        optimizer_state = None
        scheduler_state = None
        sampler_state = None
        global_step = 0
        last_relora_reset = 0
    
    # Broadcast the checkpoint info
    checkpoint_info = [global_step, last_relora_reset]
    torch.distributed.broadcast_object_list(checkpoint_info, src=0)
    global_step, last_relora_reset = checkpoint_info
    
    # Now load the base model
    print("\nLoading base model...")
    cleanup_memory()
    torch.cuda.empty_cache()
    
    # Load base model with memory-efficient settings
    model = AutoModelForCausalLM.from_pretrained(
        args.grown_model,
        device_map='cpu',
        low_cpu_mem_usage=True,
        offload_folder="offload",
        offload_state_dict=True
    )
    cleanup_memory()
    torch.cuda.empty_cache()
    
    # Enable gradient checkpointing before wrapping
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cleanup_memory()
    torch.cuda.empty_cache()
    
    # Create ReLoRA wrapper
    model = ReLoRaModel(
        model,
        target_modules=["att_proj"],
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        keep_original_weights=True,
        lora_only=False,
        trainable_scaling=False,
    )
    cleanup_memory()
    torch.cuda.empty_cache()

    # Set which parameters are trainable
    if args.do_extact_lora:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
    cleanup_memory()
    torch.cuda.empty_cache()

    # Load dataset and create dataloader first
    if args.use_on_the_fly:
        dataset = CustomBinFileDataset(
            first_idx=args.first_idx,
            last_idx=args.last_idx,
            num_tokens=args.num_tokens,
            chunk_size=args.chunk_size,
            debug=args.debug
        )
    elif args.dataset == 'dolma':
        dataset = CustomDolmaDataset(
            memmap_file=f"data/dolma_tokenized/{map_number_to_text(args.num_tokens)}.npy", 
            chunk_size=args.chunk_size, 
            debug=False, 
            num_tokens=args.num_tokens
        )
    else:
        dataset = torch.load(args.dataset)

    # Create dataloader
    sampler = ResumableSampler(dataset)
    if checkpoint is not None:
        sampler.load_state_dict(checkpoint['sampler_state'])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=1,
        pin_memory=True
    )
    print("The dataloader is successfully created.")

    # Calculate training parameters
    total_batches = len(dataloader)  # This will now be correct because sampler.__len__ accounts for start_idx
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
    args.relora_steps = cycle_length // num_gpus
    args.warmup_steps = warmup_steps
    args.restart_warmup_steps = warmup_steps // 2
    args.cycle_length = cycle_length

    # Load LoRA parameters if they exist
    if lora_state_dict is not None:
        print("Loading LoRA parameters...")
        cleanup_memory()
        torch.cuda.empty_cache()
        
        unwrapped_model = accelerator.unwrap_model(model)
        for key in list(lora_state_dict.keys()):
            unwrapped_model.state_dict()[key].copy_(lora_state_dict[key])
            del lora_state_dict[key]
            torch.cuda.empty_cache()
            
        cleanup_memory()
        torch.cuda.empty_cache()
        
        # Create optimizer and scheduler first
        optimizer = torch.optim.AdamW(
            [p for n, p in unwrapped_model.named_parameters() if p.requires_grad],
            lr=args.lr
        )
        
        start_step = global_step + 1
        effective_step = start_step - last_relora_reset
        
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
        
        # Load optimizer and scheduler states
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        if sampler_state:
            sampler.load_state_dict(sampler_state)
            
        # Clear optimizer and scheduler states
        del optimizer_state
        del scheduler_state
        del sampler_state
        cleanup_memory()
        torch.cuda.empty_cache()
        logger.info(f"Resuming from step {start_step}")
    else:
        start_step = 0
        logger.info("Starting training from beginning")
        # Create optimizer and scheduler for fresh start
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=args.lr
        )
        scheduler = get_scheculer(
            optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=steps_per_gpu * num_gpus,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            cycle_length=cycle_length,
            restart_warmup_steps=warmup_steps // 2,
        )

    # Now move model to GPU
    print("Moving model to GPU...")
    cleanup_memory()
    torch.cuda.empty_cache()
    
    model = model.to(device)
    cleanup_memory()
    torch.cuda.empty_cache()

    # print trainable parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Train ratio: {trainable_params / total_params:.4f}")

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}, shape = {param.shape}")

    # Prepare for accelerator
    print("\nPreparing for accelerator...")
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
            name=args.wandb_run_name,
            resume="allow",
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
                "resumed_from_step": start_step if lora_state_dict is not None else 0
            }
        )

    # Train model
    train_relora_preempt(model, accelerator, dataloader, optimizer, scheduler, args, 
                 checkpoint_manager=checkpoint_manager, start_checkpoint=checkpoint, sampler=sampler)

    

if __name__ == '__main__':
    main()