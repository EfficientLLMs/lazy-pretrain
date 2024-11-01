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
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


PYTHIA_TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")


class CustomBinFileDataset(Dataset):
    def __init__(self, first_idx, last_idx, num_tokens, chunk_size, debug=False):
        self.chunk_size = chunk_size
        self.collect_tokens(first_idx, last_idx, num_tokens)
        self.debug = debug
        
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
        return self.total_tokens // self.chunk_size

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = min(start + self.chunk_size, self.total_tokens)
        chunk = np.zeros(end - start, dtype=np.uint16)
        chunk_start = 0
        for mmap, length in zip(self.file_maps, self.file_lengths):
            if start < length:
                chunk_end = min(end - start, length - start)
                chunk[chunk_start:chunk_end] = mmap[start:start + chunk_end - chunk_start]
                chunk_start = chunk_end
                if chunk_start == self.chunk_size:
                    break
            start = max(0, start - length)
            end = max(0, end - length)
        # Convert uint16 to int64 before creating the PyTorch tensor
        input_ids = torch.from_numpy(chunk.astype(np.int64))

        # Print the detokenized text for debugging
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


def train(model, accelerator, dataloader, optimizer, output_dir, debug=False):
    
    model.train()
    total_loss = 0.0
    total_ppl = 0.0

    batch_loss = 0.0
    batch_ppl = 0.0

    if accelerator.is_main_process:
        print("Start training...")

    for step, batch in enumerate(tqdm(dataloader)):
        if accelerator.is_main_process:
            if step % 100 == 0:
                print(f"Training step {step}, loss: {batch_loss:.2f}, ppl: {batch_ppl:.2f}")


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

