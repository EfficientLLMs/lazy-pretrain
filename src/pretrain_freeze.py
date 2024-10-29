import argparse
from datasets import load_dataset
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import Accelerator
import wandb
import os
import gc
import json
from tqdm import tqdm
import pickle
from torch.nn.utils import clip_grad_norm_
from eval_llm import evaluate_model

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

    # parser.add_argument( "--tasks", type=str, nargs="+", 
    #                     required=True, default=["lambada_openai", "arc_easy"])

    # parser.add_argument("--num_fewshot", type=int, default=0)

    # parser.add_argument("--eval_results_path", type=str, 
    #                     default="eval/evaluation_results")

    # parser.add_argument("--parallelize", action="store_true")

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for pretraining and eval')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for pretraining')
    
    parser.add_argument('--output_dir', type=str, default='models/pythia-70m-to-pythia-410m-lora',
                        help='Directory to save the trained model')
    
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Set seed
    seed_all(args.seed)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.grown_model)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Copied layers pickle
    copied_layers_file = os.path.join(args.grown_model, "copied_layers.pkl")
    with open(copied_layers_file, "rb") as f:
        copied_layers = pickle.load(f)

    # Accelerator
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    print(f"device: {device}")

    # Freeze appropriate layers
    for i in range(len(copied_layers)):
        if not copied_layers[i]:
            parent_module = model.gpt_neox.layers[i]

            # Freeze the weights of the parent module
            for name, param in parent_module.named_parameters():
                param.requires_grad = False
        else:
            parent_module = model.gpt_neox.layers[i]

            # Freeze the weights of the parent module
            for name, param in parent_module.named_parameters():
                param.requires_grad = True
            

    # Enable gradient checkpointing (for memory efficiency)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Load dataset
    dataset = torch.load(args.dataset)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=default_data_collator,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Prepare for accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Train model
    train(model, accelerator, dataloader, optimizer, args.output_dir)

    # Delete the model to free up memory
    # First move the model to CPU
    model = model.to('cpu')
    del model

    # Clear up GPU and memory
    torch.cuda.empty_cache()
    gc.collect()
    

    

if __name__ == '__main__':
    main()