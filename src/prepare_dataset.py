import torch
import json
from transformers import AutoTokenizer

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


if __name__ == "__main__":
    dataset_path = "data/train_00/00_5m.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    dataset = prepare_dataset(dataset_path, tokenizer)
    print(f"Prepared dataset with {len(dataset)} blocks")
    print(dataset[0])

    # Save the dataset to disk
    torch.save(dataset, "data/train_00/00_5m.pt")
        
            
        