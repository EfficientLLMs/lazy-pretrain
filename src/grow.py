import argparse
import os
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import grow_depth, grow_width


MODEL_MAP = {
    'pythia-70m': 'EleutherAI/pythia-70m',
    'pythia-160m': 'EleutherAI/pythia-160m',
    'pythia-410m': 'EleutherAI/pythia-410m',
    'pythia-1.4b': 'EleutherAI/pythia-1.4b',
    'pythia-2.8b': 'EleutherAI/pythia-2.8b',
}


def grow_model(model, tokenizer, new_depth, new_width, depth_growth):

    # Save a dictionary of model's param to whether it was copied from the small model
    # This is useful for freezing the small model layers during pretraining
    copied_params = {name: False for name, _ in model.named_parameters()}

    # Grow depth
    if new_depth > model.config.num_hidden_layers:
        
        # if model depth is 6 and new depth is greater than 12,
        # grow depth in n steps of doubling the depth

        while model.config.num_hidden_layers < new_depth:
            new_layers = min(model.config.num_hidden_layers * 2, new_depth)
            print(f'Expanding model of {model.config.num_hidden_layers} layers to {new_layers} layers')
            model = grow_depth.expand_layers(
                model, model.config.num_hidden_layers, new_layers, expand_type=depth_growth
            )



    # Grow width
    if new_width > model.config.hidden_size:
        
        # Ensure that the new width = 2^n * old width
        assert math.log2(new_width // model.config.hidden_size).is_integer(), 'Width must be a power of 2'

        # Grow width in n steps of doubling the width
        while model.config.hidden_size < new_width:
            new_dim = min(model.config.hidden_size * 2, new_width)
            print(f'Expanding model of {model.config.hidden_size} width to {new_dim} width')
            model = grow_width.expand_width(
                model, model.config.hidden_size, new_dim, attn_heads=None
            )


    return model


def parse_args():

    parser = argparse.ArgumentParser(description='Grow and continue pretraining a model')

    parser.add_argument('--small_model', type=str, default='pythia-70m',
                        choices=MODEL_MAP.keys(), help='Small model to grow from')

    parser.add_argument('--large_depth', type=int, default=24,
                        help='Desired depth of the large model')
    
    parser.add_argument('--large_width', type=int, default=1024,
                        help='Desired width of the large model')
    
    parser.add_argument('--depth_growth', type=str, default='alternate',
                        choices=['alternate', 'append'], help='Depth growth strategy')

    parser.add_argument('--output_dir', type=str, default='models/pythia-70m-to-pythia-410m',
                        help='Directory to save the grown model')
    
    parser.add_argument('--checkpoint_step', type=int, default="step3000",
                        help='Model checkpoint to load')
    

    args = parser.parse_args()
    return args
    


def main():
    

    # Parse arguments
    args = parse_args()

    # Load universal tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.small_model])

    # Load small model
    small_model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.small_model],
        revision=args.checkpoint_step
    )

    # Grow small model to large model
    large_model = grow_model(
        small_model,
        tokenizer,
        args.large_depth,
        args.large_width,
        args.depth_growth
    )

    # Save the grown model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    large_model.save_pretrained(args.output_dir)

    # TODO: dict of copied params to be saved in the output_dir




if __name__ == '__main__':
    main()