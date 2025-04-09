import numpy as np
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import torch
from transformers import AutoTokenizer
import logging
import os
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from pathlib import Path
import json
import gc
import wandb

PYTHIA_TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
PYTHIA_TOKENIZER.pad_token = PYTHIA_TOKENIZER.eos_token

OLMO_TOKENIZER = AutoTokenizer.from_pretrained('allenai/OLMo-1B', trust_remote_code=True)
OLMO_TOKENIZER.pad_token = OLMO_TOKENIZER.eos_token



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDolmaDataset(Dataset):
    
    def __init__(self, memmap_file, chunk_size, num_tokens, debug=False) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.pad_token_id = OLMO_TOKENIZER.pad_token_id
        self.num_tokens = num_tokens
        self.debug = debug
        
        # Create a single memory-mapped file that stays open
        self.memmap = np.memmap(
            memmap_file,
            dtype=np.uint16,
            mode='r',
            shape=(num_tokens,)
        )
        
        if torch.cuda.is_available():
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    def __len__(self):
        return self.num_tokens // self.chunk_size

    def __getitem__(self, idx):
        if torch.cuda.is_available():
            print(f"\nBefore loading chunk {idx}, GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Calculate the start and end positions
        start = idx * self.chunk_size
        end = min(start + self.chunk_size, self.num_tokens)
        
        # Create input_ids tensor directly with padding
        input_ids = torch.full((self.chunk_size,), self.pad_token_id, dtype=torch.int64)
        
        if self.debug:
            print(f'idx: {idx}, start: {start}, end: {end}')
            print(f'Decoded OLMO_TOKENIZER.pad_token_id: {OLMO_TOKENIZER.decode([self.pad_token_id])}')
    
        # Copy the data directly from the memory-mapped file
        input_ids[:end-start] = torch.from_numpy(self.memmap[start:end])
        
        if torch.cuda.is_available():
            print(f"After copying data, GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
        if self.debug:
            print("\n\nDetokenized text:")
            print(input_ids)
            print(OLMO_TOKENIZER.decode(input_ids))
            print()

        return {"input_ids": input_ids}

# Dataset
class CustomBinFileDataset(Dataset):
    def __init__(self, first_idx, last_idx, num_tokens, chunk_size, debug=False):
        self.chunk_size = chunk_size
        self.collect_tokens(first_idx, last_idx, num_tokens)
        self.debug = debug
        self.pad_token_id = PYTHIA_TOKENIZER.pad_token_id
        
    def collect_tokens(self, first_idx, last_idx, num_tokens):
        self.file_maps = []
        self.file_lengths = []
        total_tokens = 0
        for idx in range(last_idx, first_idx - 1, -1):
            filename = f"data/pile-standard-pythia-preshuffled/document-000{idx:02d}-of-00020.bin"
            mmap = np.memmap(filename, dtype=np.uint16, mode='r')
            self.file_maps.append(mmap)
            self.file_lengths.append(len(mmap))
            total_tokens += len(mmap)
            if total_tokens >= num_tokens or idx == first_idx:
                break
        self.total_tokens = min(total_tokens, num_tokens)

    def __len__(self):
        # Round up division to include the last incomplete chunk
        return (self.total_tokens + self.chunk_size - 1) // self.chunk_size

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = min(start + self.chunk_size, self.total_tokens)
        # Initialize with padding tokens
        chunk = np.full(self.chunk_size, self.pad_token_id, dtype=np.uint16)
        
        # Calculate where we start in terms of total sequence
        total_length = sum(self.file_lengths)
        start_in_sequence = total_length - self.total_tokens + start
        end_in_sequence = total_length - self.total_tokens + end
        
        # Keep track of position in overall sequence and collect tokens
        current_pos = 0
        collected_tokens = []
        
        # Iterate through files in reverse (from earliest to latest)
        for mmap, length in zip(self.file_maps[::-1], self.file_lengths[::-1]):
            next_pos = current_pos + length
            
            # If this file contains tokens we want
            if start_in_sequence < next_pos:
                # Calculate start and end positions within this file
                file_start = max(0, start_in_sequence - current_pos)
                file_end = min(length, end_in_sequence - current_pos)
                
                # Collect tokens from this file
                tokens_to_copy = file_end - file_start
                collected_tokens.extend(mmap[file_start:file_end])
                
                if len(collected_tokens) == end - start:
                    break
                    
            current_pos = next_pos

        # Put collected tokens at the end of chunk (left padding)
        chunk[-len(collected_tokens):] = collected_tokens
        
        input_ids = torch.from_numpy(chunk.astype(np.int64))
        if self.debug:
            print("Detokenized text:")
            print(PYTHIA_TOKENIZER.decode(input_ids))
            print()
        return {"input_ids": input_ids}


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
    
    def load_latest_checkpoint(self, device='cpu'):
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
        return torch.load(checkpoint_path, map_location=device)
    
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



def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    model,
    accelerator,
    dataloader,
    optimizer,
    output_dir,
    debug=False,
    checkpoint_dir=None,
    autocast=False
):
    
    model.train()
    total_loss = 0.0
    total_ppl = 0.0

    batch_loss = 0.0
    batch_ppl = 0.0

    tokens_seen = 0

    prev_checkpoint = None

    if accelerator.is_main_process:
        print("Start training...")

    for step, batch in enumerate(tqdm(dataloader)):

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=autocast):
            outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
            loss = outputs.loss
            
        accelerator.backward(loss)


        # Clip the gradients before stepping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_loss = loss.item()
        batch_ppl = torch.exp(loss).item()

        total_loss += batch_loss
        total_ppl += batch_ppl

        tokens_seen += batch['input_ids'].numel()

        if accelerator.is_main_process:
            if step % 100 == 0:
                print(f"Training step {step}, loss: {batch_loss:.2f}, ppl: {batch_ppl:.2f}")

                # Save checkpoint
                if checkpoint_dir is not None:

                    # Delete previous checkpoint directory
                    if prev_checkpoint is not None:
                        
                        prev_checkpoint_path = checkpoint_dir + "/" + str(prev_checkpoint)
                        
                        for file in os.listdir(prev_checkpoint_path):
                            os.remove(os.path.join(prev_checkpoint_path, file))
                        os.rmdir(prev_checkpoint_path)

                        print(f"Deleted previous checkpoint {prev_checkpoint}")
                    
                    prev_checkpoint = tokens_seen
                    print(f"Saving checkpoint {prev_checkpoint} to directory {checkpoint_dir}")
                    model.module.save_pretrained(checkpoint_dir + "/" + str(prev_checkpoint))

                    print(f"Saved checkpoint to {checkpoint_dir}/{prev_checkpoint}")

        if debug and step > 5:
            break

    if debug:
        print("Debug mode is on. Training stopped early.")
        avg_loss = total_loss / 5
        avg_ppl = total_ppl / 5

    else:
        avg_loss = total_loss / len(dataloader)
        avg_ppl = total_ppl / len(dataloader)

    if accelerator.is_main_process:
        print(f"Training finished. Average loss: {avg_loss:.2f}, Average PPL: {avg_ppl:.2f}")

    
    # Save model
    if accelerator.is_main_process and not debug:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )


def cleanup_memory():
    """Clean up GPU memory and print memory usage before and after cleanup"""
    if torch.cuda.is_available():
        # Run nvidia-smi to show memory usage before cleanup (filtering for memory usage)
        # print("Before cleanup:")
        # subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.used,memory.total', '--format=csv'])

        # Empty cache and reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # Run nvidia-smi again to show memory usage after cleanup
        # print("\nAfter cleanup:")
        # subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.used,memory.total', '--format=csv'])

    # Collect garbage to free up memory
    gc.collect()


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train_full_preempt(model, accelerator, dataloader, optimizer, args, 
                 checkpoint_manager=None, start_checkpoint=None, sampler=None):
    """
    Training loop for full parameter training (with preempting).
    """

    cleanup_memory()
    model.train()

    if start_checkpoint is not None:
        start_step = start_checkpoint['global_step'] + 1
        # last_relora_reset = start_checkpoint['last_relora_reset']

        print(f"Resuming training from step {start_step}")
        
        optimizer.zero_grad()
        cleanup_memory()
    else:
        start_step = 0
        # last_relora_reset = 0
    
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
            
            # Regular training step
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
            loss = outputs.loss
            
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Get loss values and log
            loss_value = accelerator.gather(loss).mean().item()
            ppl_value = torch.exp(torch.tensor(loss_value)).item()
            current_lr = optimizer.param_groups[0]['lr']

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
                    'sampler_state': sampler.state_dict() if sampler else None,
                    'loss': loss_value,
                }
                checkpoint_manager.save_checkpoint(checkpoint_state, step)

            # Free up memory
            del loss, outputs, batch
            cleanup_memory()

    except Exception as e:
        print(f"Error during training step {step}: {str(e)}")
        if accelerator.is_main_process and checkpoint_manager is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_state = {
                'global_step': step,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sampler_state': sampler.state_dict() if sampler else None,
                'loss': loss_value,
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


if __name__ == "__main__":
    
    dataset = CustomDolmaDataset(
        memmap_file="data/dolma_tokenized/5m.npy", 
        chunk_size=2048, 
        debug=True, 
        num_tokens=5_000_000
    )

    dataset[1]
    # dataset[100]