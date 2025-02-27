import argparse
import os
import sys
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import grow_depth, grow_width_hyper
import pickle
from utils import MODEL_MAP, OLMo_1B, OLMo_7B

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.pretrain.tinyolmo import TinyOLMo




def grow_model(model, new_depth, new_width, depth_growth, attn_heads):

    # Save a dictionary of model's param to whether it was copied from the small model
    # This is useful for freezing the small model layers during pretraining
    copied_layers = [False] * model.config.num_hidden_layers

    # Grow depth
    if new_depth > model.config.num_hidden_layers:
        
        # if model depth is 6 and new depth is greater than 12,
        # grow depth in n steps of doubling the depth

        while model.config.num_hidden_layers < new_depth:
            new_layers = min(model.config.num_hidden_layers * 2, new_depth)
            print(f'Expanding model of {model.config.num_hidden_layers} layers to {new_layers} layers')
            model, copied_layers = grow_depth.expand_layers(
                model, model.config.num_hidden_layers, new_layers, expand_type=depth_growth, copied_layers=copied_layers,
            )



    # Grow width
    if new_width > model.config.hidden_size:
        
        # Ensure that the new width = 2^n * old width
        assert math.log2(new_width // model.config.hidden_size).is_integer(), 'Width must be a power of 2'

        # Grow width in n steps of doubling the width
        while model.config.hidden_size < new_width:
            new_dim = min(model.config.hidden_size * 2, new_width)
            print(f'Expanding model of {model.config.hidden_size} width to {new_dim} width')
            model = grow_width_hyper.expand_width(
                model, model.config.hidden_size, new_dim, 
            )
            # attn_heads=attn_heads
    

    return model, copied_layers


def parse_args():

    parser = argparse.ArgumentParser(description='Grow and continue pretraining a model')

    parser.add_argument('--small_model', type=str, default='OLMo-1B',
                        help='Small model to grow from')

    parser.add_argument('--large_depth', type=int, default=OLMo_7B.layers,
                        help='Desired depth of the large model')
    
    parser.add_argument('--large_width', type=int, default=OLMo_7B.hidden_size,
                        help='Desired width of the large model')
    
    parser.add_argument('--depth_growth', type=str, default='alternate',
                        choices=['alternate', 'append'], help='Depth growth strategy')

    parser.add_argument('--output_dir', type=str, default='models/OLMo-1B-to-OLMo-7B',
                        help='Directory to save the grown model')
    
    parser.add_argument('--checkpoint_step', type=str, default=None,
                        help='Model checkpoint to load')

    parser.add_argument('--attn_heads', type=int, default=None,
                        help='Number of attention heads for the large model')
    

    args = parser.parse_args()
    return args
    


def main():
    

    # Parse arguments
    args = parse_args()

    # Load universal tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-1B', trust_remote_code=True)

    # Load small model

    if 'tiny' in args.small_model:
        small_model = TinyOLMo(args.small_model)

    elif args.checkpoint_step is not None:
        small_model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAP[args.small_model],
            revision=args.checkpoint_step,
            trust_remote_code=True
        )
    else:
        small_model = AutoModelForCausalLM.from_pretrained(
            MODEL_MAP[args.small_model],
            trust_remote_code=True
        )

    PROMPT = "Finish the following sentence:\nRaindrops on roses"
    PROMPT = "Here are all the emotions written backwards:"
    
    print(f"Original model: {small_model.config.n_layers} layers, {small_model.config.d_model} width")

    # Print all parameters
    for name, param in small_model.named_parameters():
        print(f'{name}: {param.shape}')
    print(f'Model class: {type(small_model)} {small_model}')

    inputs = tokenizer(PROMPT, return_tensors="pt", return_token_type_ids=False)
    # tokens = small_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    tokens = small_model.generate(**inputs)
    # print(tokens)
    print(tokenizer.decode(tokens[0]))
    print()


    # Grow small model to large model
    large_model, copied_layers = grow_model(
        small_model,
        args.large_depth,
        args.large_width,
        args.depth_growth,
        args.attn_heads,
    )

    print(f"Grown model: {large_model.config.n_layers} layers, {large_model.config.d_model} width")
    
    # tokens = large_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    tokens = large_model.generate(**inputs)
    # print(tokens)
    print(tokenizer.decode(tokens[0]))
    print()


    # Save the grown model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    large_model.save_pretrained(args.output_dir)
    # Save copied_layers into the output_dir using json
    with open(os.path.join(args.output_dir, 'copied_layers.pkl'), 'wb') as f:
        pickle.dump(copied_layers, f)


    # Benchmark grown model
    # from eval_llm import evaluate_model
    # results = evaluate_model(
    #     model_path=args.output_dir,
    #     tokenizer_path=MODEL_MAP[args.small_model],
    #     tasks=['lambada_openai'],
    #     num_fewshot=0,
    #     batch_size=4,
    #     parallelize=True,
    # )

    # print(f"Results: {results['results']}")

    # TODO: dict of copied params to be saved in the output_dir

    print(f'Grown model config: {large_model.config}')


    # Freeze all the layers that are copied from the small model (and embedding layers)




if __name__ == '__main__':
    main()