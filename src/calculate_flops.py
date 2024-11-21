from calflops import calculate_flops
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate FLOPs of a model')
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-410m',
                        help='Model to calculate FLOPs for')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for calculating FLOPs')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Max sequence length for calculating FLOPs')
    args = parser.parse_args()
    return args




if __name__ == '__main__':

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(args.batch_size, args.max_seq_length),
        transformer_tokenizer=tokenizer,
    )

    print('-' * 50)
    print(f'FLOPs: {flops}')
    print(f'MACs: {macs}')
    print(f'Params: {params}')
    print('-' * 50)