import torch
import json
from transformers import AutoTokenizer
import argparse

def prepare_dataset(dataset_path, tokenizer):
    """
    Read dataset from disk as jsonl file and use the tokenizer to create a
    pre-tokenized dataset. Chunk the dataset into blocks of 512 tokens.

    Args:
        dataset_path (str): path to the dataset file
        tokenizer (transformers.Tokenizer): tokenizer to use for tokenization
    """

    all_tokens = []

    # Open the input file and read line by line
    with open(dataset_path, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            text = data['text']

            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

    dataset = []

    for i in range(0, len(all_tokens), 512):
        input_ids = all_tokens[i:i+512]

        if len(input_ids) == 512:

            dataset.append({
                "input_ids": input_ids
            })


    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare a dataset for pretraining')

    parser.add_argument('--dataset', type=str, default='data/train_00/00_20m.jsonl',
                        help='Path to the dataset file')

    parser.add_argument('--tokenizer', type=str, default='EleutherAI/pythia-70m',
                        help='Tokenizer to use for the model')
    
    parser.add_argument('--output', type=str, default='data/train_00/00_20m.pt',
                        help='Path to save the prepared dataset')

    return parser.parse_args()



if __name__ == "__main__":
    
    args = parse_args()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = prepare_dataset(args.dataset, tokenizer)

    print(f"Prepared dataset with {len(dataset)} blocks")
    print(dataset[0])

    # Save the dataset to disk
    torch.save(dataset, args.output)
        
            
        