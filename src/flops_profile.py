import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from pretrain.relora import ReLoRaModel, ReLoRaLinear
from peft import LoraConfig, get_peft_model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

class ManualFLOPSCounter:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _is_lora_module(m: torch.nn.Module) -> bool:
        # PEFT injects   lora_A / lora_B / lora_dropout / scaling attrs
        return hasattr(m, "lora_A") and hasattr(m, "lora_B")

    def _get_lora_rank(self, m: torch.nn.Module) -> int:
        """Return the rank *r* of the active LoRA adapter inside module *m*."""
        adapters = list(m.lora_A.keys())
        if not adapters:
            raise ValueError("LoRA module has no adapters")
        name = adapters[0]
        return m.lora_A[name].weight.shape[0]   # rows = r
        
    def count_flops(self, batch_size, seq_length):
        total_flops = 0
        flops_breakdown = {
            'relora': 0,
            'lora': 0,
            'original': 0,
            'embedding': 0,
            'layernorm': 0,
            'bias': 0,
            'other': 0
        }
        layer_count = {
            'relora': 0,
            'lora': 0,
            'original': 0,
            'embedding': 0,
            'layernorm': 0,
            'bias': 0,
            'other': 0
        }


        print("\nDetailed FLOPS Analysis:")
        # Count for each layer
        for name, module in self.model.named_modules():
            # ReLoRA layers
            if isinstance(module, ReLoRaLinear) and any(p.requires_grad for p in module.parameters()):
                layer_count['relora'] += 1
                in_features = module.in_features
                out_features = module.out_features
                
                if hasattr(module, 'r'):  # ReLoRA layer
                    rank = module.r
                    # Forward pass FLOPS
                    fwd_flops = batch_size * seq_length * (
                        2 * in_features * rank +  # LoRA path: xA + AB
                        2 * rank * out_features   # LoRA path: xA + AB
                    )
                    
                    # Backward pass FLOPS (only for LoRA matrices)
                    bwd_flops = batch_size * seq_length * (
                        4 * in_features * rank +    # Gradients for A
                        4 * rank * out_features     # Gradients for B
                    )
                    
                    layer_flops = fwd_flops + bwd_flops
                    flops_breakdown['relora'] += layer_flops
                    
                    print(f"\nLoRA Layer {layer_count['lora']}: {name}")
                    print(f"Input: {in_features}, Output: {out_features}, Rank: {rank}")
                    print(f"Forward FLOPS: {fwd_flops/1e9:.2f} GFLOPS")
                    print(f"Backward FLOPS: {bwd_flops/1e9:.2f} GFLOPS")
                    print(f"Layer Total: {layer_flops/1e9:.2f} GFLOPS")

                # For bias terms if they exist and are trainable
                if module.bias is not None and module.bias.requires_grad:
                    layer_count['bias'] += 1
                    # Forward: add bias to each output
                    bias_fwd_flops = batch_size * seq_length * out_features
                    # Backward: compute bias gradients 
                    bias_bwd_flops = batch_size * seq_length * out_features
                    bias_flops = bias_fwd_flops + bias_bwd_flops
                    flops_breakdown['bias'] += bias_flops
                    print(f"Bias FLOPS: {bias_flops/1e9:.2f} GFLOPS")


            # elif self._is_lora_module(module):
            #     in_f, out_f = module.in_features, module.out_features
            #     r = self._get_lora_rank(module)

            #     # forward
            #     # fwd_orig = 2 * in_f * out_f               # W₀·x
            #     fwd_lora = 2 * in_f * r + 2 * r * out_f   # B(Ax)
            #     fwd = batch_size * seq_length * (fwd_lora)

            #     # backward (only LoRA A & B, maybe bias)
            #     bwd = batch_size * seq_length * (4 * in_f * r + 4 * r * out_f)

            #     # optional bias in PEFT (only if trainable)
            #     # if getattr(module, "bias", None) is not None and module.bias.requires_grad:
            #     #     bwd += batch_size * seq_length * module.out_features
            #     #     fwd += batch_size * seq_length * module.out_features

            #     flops_breakdown["lora"] += fwd + bwd

            #     print(f"\nLoRA Layer {layer_count['lora']}: {name}")
            #     print(f"Input: {in_f}, Output: {out_f}, Rank: {r}")
            #     print(f"Total FLOPS: {(fwd + bwd)/1e9:.2f} GFLOPS")
            #     continue  # nothing else to do for this module


            # Original linear layers
            elif isinstance(module, torch.nn.Linear):
                print(f"\nLayer {name} is linear")
                layer_count['original'] += 1
                in_features = module.in_features
                out_features = module.out_features
                
                # Forward pass FLOPS
                fwd_flops = batch_size * seq_length * 2 * in_features * out_features
                # Backward pass FLOPS
                if module.weight.requires_grad:
                    print(f"Layer {name} is trainable")
                bwd_flops = 2*fwd_flops if module.weight.requires_grad else 0

                # If bias is trainable
                if module.bias is not None:
                    fwd_flops += batch_size * seq_length * out_features
                    if module.bias.requires_grad:
                        bwd_flops += 2*batch_size * seq_length * out_features
                
                layer_flops = fwd_flops + bwd_flops
                flops_breakdown['original'] += layer_flops
                
                print(f"\nOriginal Linear Layer: {name}")
                print(f"Input: {in_features}, Output: {out_features}")
                print(f"Forward FLOPs: {fwd_flops} FLOPs, Backward FLOPs: {bwd_flops} FLOPs")
                print(f"Total FLOPS: {layer_flops/1e9:.2f} GFLOPS")
            
            # Embedding layers
            elif isinstance(module, torch.nn.Embedding):
                layer_count['embedding'] += 1
                vocab_size = module.num_embeddings
                embed_dim = module.embedding_dim
                
                # Forward: lookup + copy
                fwd_flops = batch_size * seq_length * embed_dim

                # Backward: accumulate gradients
                bwd_flops = batch_size * seq_length * embed_dim if module.weight.requires_grad else 0
                
                layer_flops = fwd_flops + bwd_flops
                flops_breakdown['embedding'] += layer_flops
                
                print(f"\nEmbedding Layer: {name}")
                print(f"Vocab size: {vocab_size}, Embedding dim: {embed_dim}")
                print(f"Forward FLOPs: {fwd_flops} FLOPs, Backward FLOPs: {bwd_flops} FLOPs")
                print(f"Total FLOPS: {layer_flops/1e9:.2f} GFLOPs")

            # LayerNorm layers
            elif isinstance(module, torch.nn.LayerNorm):
                layer_count['layernorm'] += 1
                normalized_shape = module.normalized_shape[0]
                
                # Forward: mean(2N) + var(2N) + norm(N) + scale&shift(2N) = 7N
                fwd_flops = batch_size * seq_length * normalized_shape * 7

                # Backward: similar complexity
                bwd_flops = batch_size * seq_length * normalized_shape * 7 if module.weight.requires_grad else 0
                
                layer_flops = fwd_flops + bwd_flops
                flops_breakdown['layernorm'] += layer_flops
                
                print(f"\nLayerNorm Layer: {name}")
                print(f"Normalized shape: {normalized_shape}")
                print(f"Forward FLOPs: {fwd_flops} FLOPs, Backward FLOPs: {bwd_flops} FLOPs")
                print(f"Total FLOPS: {layer_flops/1e9:.2f} GFLOPS")
            
            # Other layers
            else:
                print(f"\nLayer {name} is not a recognized layer type")
                layer_count['other'] += 1
                pass
        
                # layer_count['other'] += 1
                # # Estimate FLOPS for other layers
                # in_features = getattr(module, 'in_features', None)
                # out_features = getattr(module, 'out_features', None)

                # if in_features and out_features and any(p.requires_grad for p in module.parameters()):
                    
                #     fwd_flops = batch_size * seq_length * 2 * in_features * out_features
                #     bwd_flops = fwd_flops
                #     layer_flops = fwd_flops + bwd_flops
                #     flops_breakdown['other'] += layer_flops
                #     print(f"\nOther Layer: {name}")
                #     print(f"Input: {in_features}, Output: {out_features}")
                #     print(f"Forward FLOPs: {fwd_flops} FLOPs, Backward FLOPs: {bwd_flops} FLOPs")
                #     print(f"Total FLOPS: {layer_flops/1e9:.2f} GFLOPS")

        # Calculate total flops first
        total_flops = sum(flops_breakdown.values())

        print(f'FLOPS breakdown:')
        print(flops_breakdown)


        # Print Summary
        print("\nLayer Counts:")
        for layer_type, count in layer_count.items():
            print(f"{layer_type}: {count}")

        print("\nFLOPS Breakdown:")
        for compute_type, flops in flops_breakdown.items():
            if total_flops > 0:  # Avoid division by zero
                percentage = (flops / total_flops) * 100
            else:
                percentage = 0.0
            print(f"{compute_type}: {flops/1e9:.2f} GFLOPS ({percentage:.1f}%)")

        print(f"\nTotal FLOPS: {total_flops/1e9:.2f} GFLOPS")
        return total_flops

def analyze_flops():
    parser = argparse.ArgumentParser(description='Calculate FLOPs manually')
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

    print(f"Using batch_size={args.batch_size}, max_seq_length={args.max_seq_length}")
    
    # Full model analysis
    print("\n=== Full Model Analysis ===")
    full_model = AutoModelForCausalLM.from_pretrained(args.model)
    total_params, trainable_params = count_parameters(full_model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Train ratio: {trainable_params/total_params:.4f}")
    
    counter = ManualFLOPSCounter(full_model)
    full_flops = counter.count_flops(args.batch_size, args.max_seq_length)
    
    # ReLoRA model analysis
    print("\n=== ReLoRA Model Analysis ===")
    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    lora_model = ReLoRaModel(
        base_model,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        keep_original_weights=True,
        lora_only=False,
        trainable_scaling=False,
    )
    
    total_params, trainable_params = count_parameters(lora_model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Train ratio: {trainable_params/total_params:.4f}")
    
    counter = ManualFLOPSCounter(lora_model)
    relora_flops = counter.count_flops(args.batch_size, args.max_seq_length)

    # LoRA model analysis
    print("\n=== LoRA Model Analysis ===")
    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        # target_modules=["query_key_value"],
        # target_modules=["att_proj"],
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(base_model, lora_config)

    total_params, trainable_params = count_parameters(lora_model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Train ratio: {trainable_params/total_params:.4f}")

    counter = ManualFLOPSCounter(lora_model)
    lora_flops = counter.count_flops(args.batch_size, args.max_seq_length)

    print(lora_model)

    
    # Compare
    print("\n=== Comparison ===")
    print(f"Full model FLOPS: {full_flops/1e9:.2f} GFLOPS")
    print(f"ReLoRA model FLOPS: {relora_flops/1e9:.2f} GFLOPS")
    print(f"LoRA model FLOPS: {lora_flops/1e9:.2f} GFLOPS")
    print(f"ReLoRA FLOPS ratio: {relora_flops/full_flops:.4f}")
    print(f"LoRA FLOPS ratio: {lora_flops/full_flops:.4f}")

if __name__ == '__main__':
    analyze_flops()