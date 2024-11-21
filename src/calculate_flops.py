from calflops import calculate_flops
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from pretrain.relora import ReLoRaModel

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate FLOPs of a model')
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-410m',
                        help='Model to calculate FLOPs for')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for calculating FLOPs')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Max sequence length for calculating FLOPs')
    parser.add_argument('--lora_rank', type=int, default=256,
                        help='Rank for LoRA')
    parser.add_argument('--lora_alpha', type=float, default=256,
                        help='Alpha for LoRA')
    args = parser.parse_args()
    return args


def calculate_flops_fullrank(model_name, batch_size, max_seq_length):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(batch_size, max_seq_length),
        transformer_tokenizer=tokenizer,
    )

    print('-' * 50)
    print(f'FLOPs: {flops}')
    print(f'MACs: {macs}')
    print(f'Params: {params}')
    print('-' * 50)


def calculate_flops_relora(model_name, batch_size, max_seq_length, lora_rank, lora_alpha):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add relora module
    model = ReLoRaModel(
        model,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        # it doesn't have these parameters
        # bias="none",  
        # task_type="CAUSAL_LM",
        keep_original_weights=True,
        lora_only=False,
        trainable_scaling=False,
    )


    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(batch_size, max_seq_length),
        transformer_tokenizer=tokenizer
    )

    print('-' * 50)
    print(f'FLOPs: {flops}')
    print(f'MACs: {macs}')
    print(f'Params: {params}')
    print('-' * 50)


if __name__ == '__main__':

    args = parse_args()

    # Calculate FLOPs for full-rank model
    calculate_flops_fullrank(args.model, args.batch_size, args.max_seq_length)

    # Calculate FLOPs for ReLoRA model
    calculate_flops_relora(args.model, args.batch_size, args.max_seq_length, args.lora_rank, args.lora_alpha)

    