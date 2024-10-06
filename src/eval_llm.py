from lm_eval import evaluator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def evaluate_merged_model(
    base_model_path: str,
    lora_path: str,
    tasks: list,
    tokenizer_path: str = None,  # If different from base_model_path
    num_fewshot: int = 0,
    batch_size: int = 4,
):
    """
    Simple evaluation of a model after merging LoRA weights.
    """
    
    # Load tokenizer
    tokenizer_path = tokenizer_path or base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
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
            model_args=f"pretrained={tmp_dir}",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            # no_cache=True
        )
    
    return results


if __name__ == "__main__":
    # Example usage
    results = evaluate_merged_model(
        base_model_path="models/pythia-70m-to-pythia-410m",
        lora_path="models/pythia-70m-to-pythia-410m-lora",
        tokenizer_path="EleutherAI/pythia-70m",  # Your HF tokenizer
        tasks=["lambada_openai", "arc_easy"],
        batch_size=4
    )
    
    # Print results
    print(results)

    # Save into a file without json serialization
    with open("eval/evaluation_results.txt", "w") as f:
        f.write(str(results))
    
    # # Optionally save results
    # import json
    # with open("evaluation_results.json", "w") as f:
    #     json.dump(results, f, indent=2)
