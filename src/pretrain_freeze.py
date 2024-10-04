import argparse
from datasets import load_dataset
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset_sample(dataset, num_samples, seed):
    # Load dataset
    dataset = load_dataset(dataset)

    print(f'Dataset size: {len(dataset["train"])}')
    
    # Sample dataset
    dataset_sample = dataset['train'].shuffle(seed=seed).select(range(num_samples))
    
    return dataset_sample


def parse_args():

    parser = argparse.ArgumentParser(description='Continue pretraining a grown model')

    parser.add_argument('--grown_model', type=str, default='models/pythia-70m-to-pythia-410m',
                        help='Path to the model to continue pretraining')

    parser.add_argument('--tokenizer', type=str, default='EleutherAI/pythia-70m',
                        help='Tokenizer to use for the model')
    
    parser.add_argument('--dataset', type=str, default='CarperAI/pile-v2-small-filtered',
                        help='Directory/hf name containing the pretraining data')
    
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to pretrain on')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for sampling the dataset')
    
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.grown_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Freeze the original model layers
    

if __name__ == '__main__':
    main()