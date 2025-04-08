import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np

OLMO_TOKENIZER = AutoTokenizer.from_pretrained(
    'allenai/OLMo-1B',
    trust_remote_code=True
)


def map_number_to_text(x):
    
    # If number in billions, return "{x}b"
    if x >= 1_000_000_000:
        return f"{x // 1_000_000_000}b"

    # If number in millions, return "{x}m"
    if x >= 1_000_000:
        return f"{x // 1_000_000}m"

    # If number in thousands, return "{x}k"
    if x >= 1_000:
        return f"{x // 1_000}k"
    
    return str(x)


def map_text_to_number(x):

    # If number in billions, return "{x}b"
    if x[-1] == 'b':
        return int(x[:-1]) * 1_000_000_000

    # If number in millions, return "{x}m"
    if x[-1] == 'm':
        return int(x[:-1]) * 1_000_000

    # If number in thousands, return "{x}k"
    if x[-1] == 'k':
        return int(x[:-1]) * 1_000
    
    return int(x)


def prepare_memmap(num_tokens, chunk_size, data_dir, output_dir, debug=False):
    
   
    dataset = load_dataset(
        'json',
        data_files=f'{data_dir}/*.json.gz',
        split='train',
        streaming=True
    )

    # dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

    # Prepare memmap
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    memmap_path = os.path.join(output_dir, f'{map_number_to_text(num_tokens)}.npy')
    memmap = np.memmap(
        memmap_path,
        dtype=np.uint16,  # Use uint16 for memory efficiency
        mode='w+',
        shape=(num_tokens,)
    )

    total_token_count = 0
    slice_token_count = 0
    all_tokens = []
    memmap_start = 0
    memmap_end = 0
    

    # If num_tokens > 20m, need to periodically flush the memmap

    with tqdm(total=num_tokens) as pbar:

        if debug:
            print(f'dataset: {dataset}')

        for data in dataset:
            text = data['text']

            if debug:
                print(f'Text: {text}')

            # If len(text) > chunk_size, split into chunks
            if len(text) > chunk_size:
                for i in range(0, len(text), chunk_size):
                    tokens = OLMO_TOKENIZER(text[i:i+chunk_size])['input_ids']
                    all_tokens.extend(tokens)
                    total_token_count += len(tokens)
                    slice_token_count += len(tokens)

                    pbar.update(len(tokens))
            else:
                tokens = OLMO_TOKENIZER(text)['input_ids']
                all_tokens.extend(tokens)
                total_token_count += len(tokens)
                slice_token_count += len(tokens)

                pbar.update(len(tokens))

            if debug:
                print(f'Tokens: {tokens}')
            
            # Flush the memmap if slice_token_count >= 20m
            if slice_token_count >= 20_000_000:
                memmap_end = memmap_start + len(all_tokens)
                memmap[memmap_start:memmap_end] = np.array(all_tokens[:num_tokens - memmap_start], dtype=np.uint16)
                memmap_start = memmap_end
                slice_token_count = 0
                all_tokens = []

            # Break
            if total_token_count >= num_tokens:
                break
        
        if debug:
            print('Done!')

    # Flush the remaining tokens, reshape the memmap if necessary    
    memmap_end = memmap_start + len(all_tokens)
    memmap[memmap_start:memmap_end] = np.array(all_tokens[:num_tokens - memmap_start], dtype=np.uint16)

    memmap.flush()


    print(f'Memmap shape: {memmap.shape}')

    # Flush
    memmap.flush()

    new_memmap = np.memmap(
        memmap_path,
        dtype=np.uint16,  # Use uint16 for memory efficiency
        mode='r',
        shape=(num_tokens,)
    )

    print(f'New memmap shape: {new_memmap.shape}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/dolma')
    parser.add_argument('--num_tokens', type=int, default=5_000_000)
    parser.add_argument('--chunk_size', type=int, default=2048)
    parser.add_argument('--output_dir', type=str, default='data/dolma_tokenized')
    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    prepare_memmap(args.num_tokens, args.chunk_size, args.data_dir, args.output_dir, debug=False)
