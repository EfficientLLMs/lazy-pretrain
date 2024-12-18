import numpy as np
import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import Accelerator
import wandb
import os
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


PYTHIA_TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
PYTHIA_TOKENIZER.pad_token = PYTHIA_TOKENIZER.eos_token


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
    checkpoint_dir=None
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

