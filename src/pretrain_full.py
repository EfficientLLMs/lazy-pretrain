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


class CustomBinFileDataset(Dataset):
    def __init__(self, first_idx, last_idx, num_tokens, chunk_size):
        self.chunk_size = chunk_size
        self.collect_tokens(first_idx, last_idx, num_tokens)
        
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
        return {"input_ids": torch.from_numpy(chunk.astype(np.int64))}


def train(model, accelerator, dataloader, optimizer, output_dir):
    
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

    avg_loss = total_loss / len(dataloader)
    avg_ppl = total_ppl / len(dataloader)

    if accelerator.is_main_process:
        print(f"Training finished. Average loss: {avg_loss:.2f}, Average PPL: {avg_ppl:.2f}")

    
    # Save model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )




def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.grown_model)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")

    # Add lora module to model
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        # target_modules=["query_key_value"],
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Add lora module to model
    model = get_peft_model(model, config)
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

    # sample = dataset[0]
    # print(f"  Shape: {sample['input_ids'].shape}")
    # print(f"  Data type: {sample['input_ids'].dtype}")
    # print(f"  First few tokens: {sample['input_ids'][:10]}")

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

    # Prepare for accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Train model
    train(model, accelerator, dataloader, optimizer, args.output_dir)

    

if __name__ == '__main__':
    main()