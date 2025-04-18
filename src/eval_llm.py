from lm_eval import evaluator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from accelerate import Accelerator
import huggingface_hub
from pretrain.utils import CheckpointManager
import tempfile
from pretrain.relora import ReLoRaModel

def evaluate_model(
    model_path: str,
    checkpoint: str,
    tasks: list,
    tokenizer_path: str = None,  # If different from base_model_path
    num_fewshot: int = 0,
    batch_size: int = 4,
    parallelize: bool = False,
):
    """
    Simple evaluation of a model.
    """

    accelerator = Accelerator()
    
    # Load tokenizer
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    if checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=checkpoint,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print(f'Loading model from {model_path}')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    # Prepare model for evaluation
    model = accelerator.prepare(model)
    
    # Save model to temporary directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Saving model to {tmp_dir}...")
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        
        # Run evaluation
        print("Starting evaluation...")
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={tmp_dir},parallelize={parallelize},logits_cache=False",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            # no_cache=True
        )
    
    return results

def evaluate_merged_model(
    base_model_path: str,
    lora_path: str,
    tasks: list,
    tokenizer_path: str = None,  # If different from base_model_path
    num_fewshot: int = 0,
    batch_size: int = 4,
    parallelize: bool = False,
):
    """
    Simple evaluation of a model after merging LoRA weights.
    """

    print("Using merged model.")
    
    # Load tokenizer
    tokenizer_path = tokenizer_path or base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load and merge LoRA weights
    print("Loading and merging LoRA weights...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    
    # Save merged model to temporary directory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Saving merged model to {tmp_dir}...")
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        
        # Run evaluation
        print("Starting evaluation...")
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={tmp_dir},parallelize={parallelize}",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            # no_cache=True
        )
    
    return results


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path", type=str, required=True, default="models/pythia-70m-to-pythia-410m"
    )
    parser.add_argument(
        '--checkpoint_step', type=str, default=None, help='Model checkpoint to load')
    parser.add_argument(
        "--lora_path", type=str, required=False, default=None
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="EleutherAI/pythia-70m",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", required=True, default=["lambada_openai", "arc_easy"]
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0
    )
    parser.add_argument(
        "--batch_size", type=int, default=4
    )

    parser.add_argument(
        "--eval_results_path", type=str, default="eval/evaluation_results"
    )

    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to the checkpoint folder to evaluate"
    )

    parser.add_argument(
        "--parallelize", action="store_true"
    )

    parser.add_argument(
        "--token", type=str, default=None, help="HuggingFace token for private dataset"
    )

    parser.add_argument('--rank', type=int, default=256,
        help='Rank of the Lora module'
    )

    parser.add_argument('--lora_alpha', type=int, default=256,
                        help='Alpha of the lora module (\\deltaW * \\alpha/r)')

    return parser.parse_args()


if __name__ == "__main__":
    
    # Parse arguments
    args = parse_args()

    print('Number of gpus:', torch.cuda.device_count())

    # Log into HuggingFace
    if args.token is not None:
        # Read token from file
        with open(args.token, 'r') as f:
            token = f.read().strip()
        # Log into HuggingFace
        huggingface_hub.login(token=token)


    

    if args.checkpoint_path is not None:
        
        checkpoint_manager = CheckpointManager(args.checkpoint_path)
        latest_checkpoint = checkpoint_manager.load_latest_checkpoint()

        # Create temp file of merged model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
            )

            # Remove wrapped_model. from state dict keys
            is_relora = False
            for k, v in latest_checkpoint['model_state_dict'].items():
                if 'lora_A' in k:
                    is_relora = True
                    continue

            # If relora weights present, create ReLora Model
            if is_relora:
                model = ReLoRaModel(
                    model,
                    target_modules=[
                        "query_key_value",
                        "dense",
                        "dense_h_to_4h",
                        "dense_4h_to_h",
                    ],
                    r=args.rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=0.05,
                    # it doesn't have these parameters
                    # bias="none",  
                    # task_type="CAUSAL_LM",
                    keep_original_weights=True,
                    lora_only=False,
                    trainable_scaling=False,
                )

            # Load state dict from latest checkpoint
            model.load_state_dict(latest_checkpoint['model_state_dict'])

            # Save model to temporary directory
            model.save_pretrained(tmp_dir)

            # Run evaluation
            results = evaluate_model(
                model_path=tmp_dir,
                checkpoint=None,
                tokenizer_path=args.tokenizer_path,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                parallelize=args.parallelize,
            )

    elif args.lora_path is None:
        results = evaluate_model(
            model_path=args.base_model_path,
            checkpoint=args.checkpoint_step,
            tokenizer_path=args.tokenizer_path,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            parallelize=args.parallelize,
        )

    else:
        results = evaluate_merged_model(
            base_model_path=args.base_model_path,
            lora_path=args.lora_path,
            tokenizer_path=args.tokenizer_path,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            parallelize=args.parallelize,
        )
    
    # Print results
    print(results['results'])

    # Save into a file without json serialization
    # with open(args.eval_results_path + "_full.txt", "w") as f:
    #     f.write(str(results))
    
    # Optionally save results
    import json
    with open(args.eval_results_path + ".json", "w") as f:
        json.dump(results['results'], f, indent=2)
