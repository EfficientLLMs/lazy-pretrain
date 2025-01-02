import os
import json
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import transformers
import wandb

from loguru import logger
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

# from peft_pretraining.modeling_llama import LlamaDecoderLayer


def initialize_fsdp(model, dtype):
    wrapping_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    if dtype in ["bf16", "bfloat16"]:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,  # Gradient communication precision
            buffer_dtype=torch.bfloat16,  # Buffer precision
        )
    elif dtype == "float32":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,  # Gradient communication precision
            buffer_dtype=torch.float32,  # Buffer precision
        )
    else:
        raise ValueError(f"Dtype {dtype} not supported (only float32 and bfloat16 are)")

    model = FSDP(
        model,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=wrapping_policy,
    )
    return model


def get_scheculer(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
):
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for cosine_restarts scheduler")

    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine_restarts":
        assert restart_warmup_steps is not None, "restart_warmup_steps must be specified for cosine_restarts scheduler"
        print("\nScheduler Configuration:")
        print(f"  num_training_steps: {num_training_steps}")
        print(f"  first_warmup_steps: {warmup_steps}")
        print(f"  restart_warmup_steps: {restart_warmup_steps}")
        print(f"  cycle_length: {cycle_length}")
        print(f"  min_lr_ratio: {min_lr_ratio}")
        print(f"  expected number of restarts: {num_training_steps // cycle_length}")
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cyclical_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, cycle_length, min_lr_ratio=0.1, last_epoch=-1):
    assert cycle_length is not None or num_training_steps is not None, "You must specify either cycle_length or num_training_steps"
    
    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})")

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    # if num_training_steps % restart_every != 0:
    #     raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def random_pruning_(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor.mul_(random_pruning_mask)


@torch.no_grad()
def magnitude_pruning_(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor.mul_(mask.to(dtype=tensor.dtype))


def _get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Global cosine decay with periodic resets.
    The LR decays from 1.0 to min_lr_ratio over entire training,
    with periodic resets and warmups.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps <= num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps <= restart_every, "the first reset will happen before the warmup is done"

    # Calculate global decay factor based on total progress
    progress = float(current_step) / float(num_training_steps)
    global_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    target_lr = min_lr_ratio + (1.0 - min_lr_ratio) * global_decay

    # Initial warmup
    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps)) * 1.0  # Warmup to full LR

    _current_step = current_step + adjust_step
    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    # Handle warmup after resets
    if restart_step < restart_warmup_steps and current_step >= restart_every:
        return float(restart_step) / float(max(1, restart_warmup_steps)) * target_lr
    
    return target_lr


def max_train_tokens_to_number(max_train_tokens):
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)


def get_last_training_state(save_dir):
    # list all directories in the save_dir
    # find the model with the highest number of iterations "{args.save_dir}/model_{update_step}"
    model_dirs = [d for d in os.listdir(save_dir) if d.startswith(f"model_")]
    if len(model_dirs) == 0:
        logger.warning(f"Save directory {save_dir} exists, but does not contain any models.")
        logger.warning("Starting training from scratch.")
        return None, None

    model_dirs = sorted(model_dirs, key=lambda x: int(x.split("_")[-1]))
    resume_from = os.path.join(save_dir, model_dirs[-1])

    logger.info(f"Restarting training from {resume_from}")
    with open(os.path.join(resume_from, "training_state.json")) as f:
        training_state = json.load(f)

    return training_state, resume_from


def optimizer_reset(
    optimizer,
    *,
    reset_params: list[torch.nn.Parameter],
    optimizer_state_keys: list[str],
    reset_optimizer_on_relora: bool,
    optimizer_random_pruning: float,
    optimizer_magnitude_pruning: float,
):
    """
        optimizer_state_keys: e.g., ["exp_avg", "exp_avg_sq"]
    """
    n_reset_types = (
        int(bool(reset_optimizer_on_relora))
        + int(bool(optimizer_random_pruning))
        + int(bool(optimizer_magnitude_pruning))
    )
    if n_reset_types != 1:
        logger.warning(f"Got {reset_optimizer_on_relora=}, {optimizer_random_pruning=}, "
                       f"{optimizer_magnitude_pruning=}")
        raise ValueError(f"Exactly one of reset_optimizer_on_relora, "
                         f"optimizer_random_pruning, optimizer_magnitude_pruning must be True")

    # pruning_fn has to be inplace to work with ZeroRedundancyOptimizer
    if reset_optimizer_on_relora:
        logger.info("Resetting optimizer states to zeros")
        # looks like zeroing out breaks dictionary in the optimizer
        # see full error below
        pruning_fn = partial(random_pruning_, prune_ratio=0.999)
    elif optimizer_random_pruning:
        logger.info(f"Performing random pruning of optimizer states. "
                    f"Pruning {optimizer_random_pruning} percent")
        pruning_fn = partial(random_pruning_, prune_ratio=optimizer_random_pruning)
    elif optimizer_magnitude_pruning:
        logger.info(f"Performing magnitude pruning of optimizer states. "
                    f"Pruning {optimizer_magnitude_pruning} percent")
        pruning_fn = partial(magnitude_pruning_, prune_ratio=optimizer_magnitude_pruning)
    else:
        raise ValueError("Unknown pruning type")

    # ############################################################
    # A reminder on how optimizer state is structured for regular optimizers:
    # optimizer.state is a dict[torch.nn.Parameter, dict[str, torch.Tensor]]
    # optimizer.state[p] is a dict[str, torch.Tensor] where str is
    # an optimizer state key e.g., "exp_avg", "exp_avg_sq"
    # Note that none of these tensors has parameter names
    # and parameter maps to a **dictionary** of opt. states, not a tensor
    # 
    # For ZeroRedundancyOptimizer, it works differently.
    # ZeroRedundancyOptimizer.state always maps to empty dicts.
    # Instead, it uses optimizer.optim.state for rank-local updates.
    # 
    # For some reason, zeroing out a tensor in ZeroRedundancyOptimizer.opt.state
    # causes an error during state_dict collection.
    # This is why we use 0.999 pruning ratio for reset_optimizer case.
    # 
    # Here's an error that happens:
    # 
    # Traceback (most recent call last):
    # File ".../peft_pretraining/torchrun_main.py", line 866, in <module>
    #     main(args)
    # File ".../peft_pretraining/torchrun_main.py", line 715, in main
    #     save_model(
    # File ".../peft_pretraining/torchrun_main.py", line 289, in save_model
    #     save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir)
    # File ".../peft_pretraining/torchrun_main.py", line 224, in save_model_ddp
    #     optimizer.consolidate_state_dict()
    # File ".../python3.10/site-packages/torch/distributed/optim/zero_redundancy_optimizer.py", line 565, in consolidate_state_dict
    #     self.optim.state_dict(),
    # File ".../python3.10/site-packages/torch/optim/optimizer.py", line 364, in state_dict
    #     packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
    # File ".../python3.10/site-packages/torch/optim/optimizer.py", line 364, in <dictcomp>
    #     packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
    # KeyError: 140580723685184
    # 
    # One one hand, the hypothesis is that making a zero tensor
    # is implementing by changing the pointer in the memory to
    # an existing zero-tensor. But on the other hand, we didn't
    # have issues with that when using regular Adam, without ZeroRedundancyOptimizer wrapper.
    # ############################################################
    n_zeros = 0
    n_total = 0

    optimizer_state = optimizer.state
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer_state = optimizer.optim.state

    for p in reset_params:
        param_state = optimizer_state[p]
        if len(param_state) == 0: # no state for this param, happens for ZeRo optimizer
            continue
        for key in optimizer_state_keys:
            pruning_fn(param_state[key])  # pruning fn has to be inplace to keep the same keys in the dict
            n_total += param_state[key].numel()
            n_zeros += torch.sum(param_state[key] == 0).item()

    _zeroed = n_zeros / (1e-7 + n_total) * 100
    logger.info(f"Percent of optimizer states zeroed: {_zeroed:.2f}")


def print_optimizer_state_size(optimizer):
    # Count the number of floats in the first and second moments
    first_moment_count = 0
    second_moment_count = 0

    optimizer_state = optimizer.state
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer_state = optimizer.optim.state

    for state in optimizer_state.values():
        if len(state) == 0: # no state for this param, happens for ZeRo optimizer
            continue

        first_moment_count += torch.numel(state['exp_avg'])
        second_moment_count += torch.numel(state['exp_avg_sq'])

    global_rank = 0
    if dist.is_initialized():
        global_rank = dist.get_rank()

    print(f"(Rank {global_rank}) Number of floats in the first moment: {first_moment_count / 1_000_000:.2f}M")
    print(f"(Rank {global_rank}) Number of floats in the second moment: {second_moment_count / 1_000_000:.2f}M")


def check_lr_and_alert(optimizer, max_lr):
    global_rank = 0 if not dist.is_initialized() else dist.get_rank()

    lr = optimizer.param_groups[0]["lr"]
    if lr <= max_lr: return

    alert_message = f"Optimizer lr after the reset is large. This can lead to instability. Current lr is {lr}"
    logger.warning(alert_message)
    if global_rank == 0:
        wandb.alert(
            title="Learning rate issue",
            text=alert_message,
            level=wandb.AlertLevel.WARN,
        )

def delete_old_checkpoints(save_dir, keep):
    if keep is None:
        return

    checkpoints = [d for d in os.listdir(save_dir) if d.startswith(f"model_")]
    if len(checkpoints) <= keep:
        return

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
    for checkpoint in checkpoints[:-keep]:
        checkpoint_path = os.path.join(save_dir, checkpoint)
        logger.info(f"Deleting checkpoint {checkpoint_path}")
        os.system(f"rm -rf {checkpoint_path}")


def train_relora(model, accelerator, dataloader, optimizer, scheduler, args, output_dir):
    """
    Training loop for ReLoRA.
    """
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


def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()


def train_relora_preempt(model, accelerator, dataloader, optimizer, scheduler, args, 
                 checkpoint_manager=None, start_checkpoint=None, sampler=None):
    """
    Training loop for ReLoRA.
    """

    cleanup_memory()
    model.train()

    if start_checkpoint is not None:
        start_step = start_checkpoint['global_step'] + 1
        last_relora_reset = start_checkpoint['last_relora_reset']

        optimizer.zero_grad()
        cleanup_memory()
    else:
        start_step = 0
        last_relora_reset = 0
    
    # Debug prints from all processes
    print(f"\nProcess {accelerator.process_index} debug info:")
    print(f"  Device: {accelerator.device}")
    print(f"  Is main process: {accelerator.is_main_process}")
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Is model training: {model.training}")
    print(f"  Starting/Resuming from step: {start_step}")

    # Synchronize before starting training loop
    print(f"\nProcess {accelerator.process_index} waiting for synchronization...")
    accelerator.wait_for_everyone()
    print(f"Process {accelerator.process_index} synchronized")

    if accelerator.is_main_process:
        print("\nStarting training loop...")
        progress_bar = tqdm(total=len(dataloader), desc="Training", initial=start_step)
    
    try:
        for step, batch in enumerate(dataloader, start=start_step):
            # ReLoRA reset logic
            next_reset_step = last_relora_reset + args.relora_steps
            
            if step >= next_reset_step and step % args.relora_steps == 0:
                if accelerator.is_main_process:
                    print(f"\nPerforming LoRA reset at step {step}")
                
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.merge_and_reinit()
                
                lora_params = [p for n, p in unwrapped_model.named_parameters() if "lora_" in n]
                optimizer_reset(
                    optimizer,
                    reset_params=lora_params,
                    optimizer_state_keys=["exp_avg", "exp_avg_sq"],
                    reset_optimizer_on_relora=True,
                    optimizer_random_pruning=0.0,
                    optimizer_magnitude_pruning=0.0,
                )
                
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    wandb.log({"relora_reset": step}, step=step)
                    print("LoRA reset complete")
                
                last_relora_reset = step

            # Regular training step
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
            loss = outputs.loss
            
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Get loss values and log
            loss_value = accelerator.gather(loss).mean().item()
            ppl_value = torch.exp(torch.tensor(loss_value)).item()
            current_lr = scheduler.get_last_lr()[0]

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
            
            # Save checkpoint
            if accelerator.is_main_process and (step > 0) and (step % args.checkpoint_freq == 0):
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_state = {
                    'global_step': step,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'sampler_state': sampler.state_dict() if sampler else None,
                    'loss': loss_value,
                    'last_relora_reset': last_relora_reset,
                    'cycles_completed': step // args.relora_steps,
                }
                checkpoint_manager.save_checkpoint(checkpoint_state, step)

    except Exception as e:
        print(f"Error during training step {step}: {str(e)}")
        if accelerator.is_main_process and checkpoint_manager is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_state = {
                'global_step': step,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'sampler_state': sampler.state_dict() if sampler else None,
                'loss': loss_value,
                'last_relora_reset': last_relora_reset,
                'cycles_completed': step // args.relora_steps,
            }
            checkpoint_manager.save_checkpoint(checkpoint_state, step)
        raise e

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Training finished.")
        progress_bar.close()
        wandb.finish()

        if not args.debug:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )