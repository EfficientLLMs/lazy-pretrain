import torch
import torch.nn as nn
import copy
from utils import MODEL_MAP
from functools import partial
import numpy as np


def wide_bias(x, old_width, new_width, average=False):
    """
    Function preserving expansion of bias vector from (old_width) 
    to (new_width)

    Args:
        x (torch.Tensor): input tensor of shape (old_width)
        old_width (int): old width of the bias vector
        new_width (int): new width of the bias vector

    Returns:
        torch.Tensor: expanded tensor of shape (new_width)
    """

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"
    assert new_width == 2 * old_width, "New width must be twice the old width"

    new_cols = (x.shape[0] * new_width) // old_width
    y = torch.zeros(new_cols)

    y[:x.shape[0]] = x
    y[-x.shape[0]:] = x

    return y

def wide_matrix(x, old_width, new_width, average=False):
    """
    Function preserving expansion of weight matrix from (old_width, old_width) 
    to (new_width, new_width)

    A weight matrix is of shape (out_size, in_size)

    Args:
        x (torch.Tensor): input tensor of shape (old_width)
        old_width (int): old width of the weight matrix
        new_width (int): new width of the weight matrix

    Returns:
        torch.Tensor: expanded tensor of shape (new_width, new_width)
    """

    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"
    assert new_width == 2 * old_width, "New width must be twice the old width"

    new_rows = (x.shape[0] * new_width) // old_width
    new_cols = (x.shape[1] * new_width) // old_width

    y = torch.zeros(new_rows, new_cols)
    eta = torch.zeros_like(x)    # Add random noise
    # eta = torch.randn_like(x) * 1e-3    # Add random noise
    
    # Copy old matrix into new matrix into 4 corners
    y[:x.shape[0], :x.shape[1]] = x/2 + eta
    y[-x.shape[0]:, -x.shape[1]:] = x/2 - eta
    y[:x.shape[0], -x.shape[1]:] = x/2 + eta
    y[-x.shape[0]:, :x.shape[1]] = x/2 - eta

    return y

def wide_attn(x, old_width, new_width, attn_ratio):
    """
    Function preserving expansion of attention layer from (old_width, old_width) 
    to (new_width, new_width)

    Args:
        x (torch.Tensor): input tensor of shape (old_width, old_width)
        old_width (int): old width of the attention layer
        new_width (int): new width of the attention layer

    Returns:
        torch.Tensor: expanded tensor of shape (new_width, new_width)
    """
    
    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"

    # expand weight normally
    y = wide_matrix(x, old_width, new_width)

    # adjust query weight alone by torch.sqrt(attn_ratio)
    # y[:new_width, :] *= torch.sqrt(torch.tensor(attn_ratio))
    y[:old_width, :] *= torch.sqrt(torch.tensor(attn_ratio))
    y[3*old_width:4*old_width, :] *= torch.sqrt(torch.tensor(attn_ratio))

    return y


     

def wide_embedding_in(x, old_width, new_width):
    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"
    assert new_width == 2 * old_width, "New width must be twice the old width"

    y = torch.zeros(x.shape[0], new_width)

    # Apply Apple HyperClone expansion
    y[:, :old_width] = x
    y[:, old_width:] = x

    return y


def wide_embedding_out(x, old_width, new_width):
    
    assert new_width >= old_width, "New width must be greater than or equal to old width"
    assert new_width - old_width <= old_width, "New width must be at most twice the old width"
    assert new_width == 2 * old_width, "New width must be twice the old width"


    eta = torch.zeros_like(x)    # Add random noise
    # eta = torch.randn_like(x) * 1e-3    # Add random noise

    new_rows = x.shape[0]
    new_cols = (x.shape[1] * new_width) // old_width
    y = torch.zeros(new_rows, new_cols)

    # Copy old matrix into new matrix
    y[:, :x.shape[1]] = x / 2 + eta

    # Copy first new_cols-x.shape[1] columns of x into the last new_cols-x.shape[1] columns of y
    y[:, x.shape[1]:] = x / 2 - eta

    return y


def wide_state_dict(old_state_dict, old_width, new_width, old_head_dim, new_head_dim):
    new_state_dict = {}
    for key, weight in old_state_dict.items():
        new_state_dict[key] = wide_param(key, weight, old_width, new_width, old_head_dim, new_head_dim)
    
    # Clone the unembedding layer from the source network embedding layer
    new_state_dict['model.transformer.ff_out.weight'] = wide_embedding_out(old_state_dict['model.transformer.wte.weight'], old_width, new_width)
    
    return new_state_dict


def reorder_weights(w, n_heads_old, head_dim_old, n_repeat_dim, n_repeat_heads):
    """
    Reorders the columns of the out_project linear layer at the end of the
    attention layer.

    This function is meant to preserve the functionality of the attention
    block when the head_dim is changed.

    Arguments:
        w:
            The weight tensor to be reordered (from the o_proj linear layer).
        n_heads_old:
            Number of heads in the source attention layer.
        head_dim_old:
            Dimension of each head in the source attention layer.
        n_repeat_dim:
            Number of times the head-dim of the source attention layer is
            repeated in the destination attention layer.
        n_repeat_heads:
            Number of times the heads of the source attention layer are
            repeated in the destination attention layer.

    Returns:
        Reordered weights.
    """
    w = w.reshape(w.shape[0], n_repeat_heads, n_repeat_dim, n_heads_old, head_dim_old)
    sh_old = copy.copy(w.shape)
    w = w.permute(0, 1, 3, 2, 4)
    sh_new = copy.copy(w.shape)
    w = w.reshape(w.shape[0], -1)
    return w


def reorder_swiglu(tensor, n_repeat):
    """
    Reorders the rows in the first linear layer of the FFN to correct
    the gating that occurs in the subsequent SWIGLU activation function.

    The original linear layer produces the output [x_top, x_bottom]^T,
    and the following SWIGLU computes the activation as 'silu(x_top) * x_bottom'.

    In contrast, the cloned layer produces the output [x_up, x_down, x_up, x_down]^T.
    The SWIGLU incorrectly computes 'silu([x_top, x_bottom]^T) * [x_top, x_bottom]^T',
    which is not the desired behavior.

    To fix this, we need to reorder the weights of the cloned linear layer
    so that it produces [x_up, x_up, x_down, x_down]^T. This allows the SWIGLU
    to correctly compute 'silu([x_top, x_top]^T) * [x_bottom, x_bottom]^T'.

    Arguments:
        tensor:
            The weight or bias tensor in the destination FFN block.
        n_repeat:
            The expansion factor from the source to the destination FFN block.

    Returns:
        Reordered tensor.
    """
    n = int(tensor.shape[0] // (n_repeat * 2))
    tensors_up = [tensor[2 * i * n : (2 * i + 1) * n] for i in range(0, n_repeat)]
    tensors_down = [
        tensor[(2 * i + 1) * n : (2 * i + 2) * n] for i in range(0, n_repeat)
    ]
    all_tensors = tensors_up + tensors_down
    return torch.cat(all_tensors, dim=0)



def clone_olmo_qkv_weight(
    dst_weight_shape, src_weight, num_heads_dst, num_heads_src
):
    """
    Clones 'src_weight' into a weight tensor with 'dst_weight_shape' to be
    used in the attention layer.

    Arguments:
        dst_weight_shape:
            Shape of the weight tensor in the destination layer.
        src_weight:
            Weight tensor in the source layer.
        num_heads_dst:
            Number of attention heads in the destination layer.
        num_heads_src:
            Number of attention heads in the source layer.
        snr_db:
            Signal-to-noise ratio. Defaults to None.

    Returns:
        Cloned QKV weights.
    """
    assert src_weight.shape[0] == (3 * src_weight.shape[1])
    assert dst_weight_shape[0] == (3 * dst_weight_shape[1])
    source_embedding_dim = src_weight.shape[1]
    destination_embedding_dim = dst_weight_shape[1]
    
    n_repeat = destination_embedding_dim // source_embedding_dim
    
    dst_weight = src_weight.reshape(
        3, num_heads_src, source_embedding_dim // num_heads_src, source_embedding_dim
    )  # (3, H, E/H, E)
    block_shape = dst_weight.shape
    head_repeat = num_heads_dst // num_heads_src
    dim_repeat = n_repeat // head_repeat
    dst_weight = (
        dst_weight.repeat(1, head_repeat, dim_repeat, n_repeat) / n_repeat
    )  # (3, nH, E/H, nE)
    dst_weight[:2] = dst_weight[:2] / np.sqrt(
        np.sqrt(dim_repeat)
    )  ##divide query and key weights to compensate for normalization
    
    dst_weight = dst_weight.reshape(
        3 * destination_embedding_dim, destination_embedding_dim
    )  # (3, n_heads, d_head, e) --> #(3*n_heads*d_head, e)

    return dst_weight


def wide_param(key, weight, old_width, new_width, old_head_dim, new_head_dim, **kwargs):
    if 'wte' in key:
        # only output dim expands
        return wide_embedding_in(weight, old_width, new_width)
    
    # elif 'embed_out' in key:
    #     # only input dim expands
    #     if 'bias' in key:
    #         return weight
    #     elif 'weight' in key:
    #         return wide_embedding_out(weight, old_width, new_width)
        
    elif 'att_proj' in key:
        # expand attention layer
        if 'bias' in key:
            return wide_bias(weight, old_width, new_width)
        elif 'weight' in key:
            # return wide_matrix(weight, old_width, new_width)
            attn_ratio = old_head_dim / new_head_dim
            # return wide_attn(weight, old_width, new_width, attn_ratio)

            return clone_olmo_qkv_weight(
                (3 * new_width, new_width),
                weight,
                new_width // new_head_dim,
                old_width // old_head_dim,
            )
        
    elif 'attn_out' in key:

        # expand attention layer
        if 'bias' in key:
            return wide_bias(weight, old_width, new_width)
        elif 'weight' in key:
            new_weight = wide_matrix(weight, old_width, new_width)

            if new_head_dim != old_head_dim:
                
                num_heads_multiplier = kwargs.get("num_heads_multiplier", new_width // old_width)
                new_weight = reorder_weights(
                    new_weight,
                    n_heads_old=old_width // old_head_dim,
                    head_dim_old=old_head_dim,
                    n_repeat_dim=new_head_dim // old_head_dim,
                    n_repeat_heads=num_heads_multiplier
                )

            return new_weight

            return wide_attn_out(weight, old_width, new_width)
        
    # elif 'final_layer_norm' in key or 'ln_f' in key:
    #     return wide_bias(weight, old_width, new_width)

    elif 'ln_f' in key or 'ff_norm' in key or 'attn_norm' in key:
        return wide_bias(weight, old_width, new_width)

    elif 'ff_proj' in key:
        # expand feedforward layer
        if 'bias' in key:
            return wide_bias(weight, old_width, new_width)
        elif 'weight' in key:
            new_weight = wide_matrix(weight, old_width, new_width)
            
            # If the input to SWIGLU activaion has changed dimension, we should reorder weights:
            swiglu_repeat = new_width // old_width

            if swiglu_repeat > 1:
                new_weight = reorder_swiglu(new_weight, swiglu_repeat)

            return new_weight
    
    elif 'weight' in key:
        return wide_matrix(weight, old_width, new_width)
    # elif 'bias' in key:
    #     return wide_bias(weight, old_width, new_width)

    else:
        print(f'key {key} getting skipped')

def expand_width(model, old_width, new_width, attn_heads=None):
    """
    Expand the width of a model in a function preserving model from size `old_width` to 
    `new_width`. 

    Args:
        model (transformers.AutoModelForCausalLM): The language model to expand
        old_width (int): The old width of the model
        new_width (int): The new width of the model

    Returns:
        model (transformers.AutoModelForCausalLM): The expanded model
    """

    # Save old model weights in state dict
    old_state_dict = model.state_dict()

    

    # Use a copy of the model to avoid changing the original model
    old_config = model.config
    new_config_dict = old_config.to_dict()

   
    # Calculate new number of attention heads as new_width / (old_width/old_n_heads)
    new_n_heads = int(new_width / (old_width / old_config.n_heads))
    
    
    # new_config_dict["hidden_size"] = new_width
    new_config_dict["d_model"] = new_width
    # new_config_dict["intermediate_size"] = new_width * 4
    new_config_dict["n_heads"] = new_n_heads
    # new_config_dict["_name_or_path"] += f"-expand-width-{new_width}"
    new_config_dict["mlp_hidden_size"] = new_width * old_config.mlp_ratio
    new_config_dict["weight_tying"] = False
    new_config = type(old_config).from_dict(new_config_dict)
    
    if hasattr(model, 'model_dir'):
        pass

    model = type(model)(new_config)

    # Set config hidden size
    if attn_heads is not None:
        model.config.n_heads = attn_heads

    old_head_dim = old_width // old_config.n_heads
    new_head_dim = new_width // new_n_heads


    # Create new state dict
    new_state_dict = wide_state_dict(old_state_dict, old_width, new_width, old_head_dim, new_head_dim)


    # Load new state dict
    model.load_state_dict(new_state_dict)

    return model
    


# hook function to view intermediate activations
def forward_hook(module, input, output, layer_name=None):
    print(f'Inside {module.__class__.__name__} forward {layer_name}')
    print(f'Input: {input}')
    print(f'Output: {output}')
    print(f'Output shape: {output.shape}')


if __name__ == '__main__':

    
    torch.manual_seed(1)
    torch.set_printoptions(profile="default")


    from transformers import AutoModelForCausalLM, AutoTokenizer


    model_1b = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP['OLMo-1B'],
        cache_dir="../.cache/OLMo-1B",
        trust_remote_code=True
    )

    tokenizer_1b = AutoTokenizer.from_pretrained(
        MODEL_MAP['OLMo-1B'],
        cache_dir="../.cache/OLMo-1B",
        trust_remote_code=True
    )

    print('Model 1B config')
    print(model_1b.config)
    # print('Model 1B model')
    # print(model_1b.model.config)
    print('Model 1B')
    print(model_1b)

    # Print all model parameters
    for name, param in model_1b.named_parameters():
        print(name)


    PROMPT = "Finish the following sentence:\nRaindrops on roses"
    PROMPT = "Finish the following sentence:\nHappy birthday in gibberish is"


    # Register forward_hook on model_1b.model.transformer.wte
    # model_1b.model.transformer.blocks[0].attn_out.register_forward_hook(partial(forward_hook, layer_name='attn_out'))
    # model_1b.model.transformer.blocks[0].ff_out.register_forward_hook(partial(forward_hook, layer_name='ff_out'))
    # model_1b.model.transformer.blocks[0].att_proj.register_forward_hook(partial(forward_hook, layer_name='att_proj'))
    # model_1b.model.transformer.blocks[0].ff_proj.register_forward_hook(partial(forward_hook, layer_name='ff_proj'))
    # model_1b.model.transformer.ln_f.register_forward_hook(partial(forward_hook, layer_name='ln_f'))

    print(f"Original model: {model_1b.config.n_layers}")
    inputs = tokenizer_1b(PROMPT, return_tensors="pt")
    tokens = model_1b.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    # print(f'Raw logits: {output.logits[0]}')
    # tokens = output.logits[0].argmax(dim=-1)
    print(f'Generated tokens: {tokens[0]}')
    print(tokenizer_1b.decode(tokens[0]))
    # print(f"hidden state: {output[0].shape}")

    # Output of single forward pass:
    # model_output = model_1b(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    # print(f"Model output: {model_output}")

    model_1b_expanded = expand_width(model_1b, 2048, 4096)

    # Register forward_hook on model_1b.model.transformer.wte
    # model_1b_expanded.model.transformer.blocks[0].attn_out.register_forward_hook(partial(forward_hook, layer_name='attn_out'))
    # model_1b_expanded.model.transformer.blocks[0].ff_out.register_forward_hook(partial(forward_hook, layer_name='ff_out'))
    # model_1b_expanded.model.transformer.blocks[0].att_proj.register_forward_hook(partial(forward_hook, layer_name='att_proj'))
    # model_1b_expanded.model.transformer.blocks[0].ff_proj.register_forward_hook(partial(forward_hook, layer_name='ff_proj'))
    # model_1b_expanded.model.transformer.ln_f.register_forward_hook(partial(forward_hook, layer_name='ln_f'))

    print(f"Expanded model: {model_1b_expanded.config}")

    

    inputs = tokenizer_1b(PROMPT, return_tensors="pt")
    # output = model_1b_expanded(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    tokens = model_1b_expanded.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    # print(f'Raw logits: {output.logits[0]}')
    # tokens = output.logits[0].argmax(dim=-1)
    print(f'Generated tokens: {tokens[0]}')
    print(tokenizer_1b.decode(tokens[0]))
    # print(f"hidden state: {output[0].shape}")

    # Output of single forward pass:
    # model_output = model_1b_expanded(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    # print(f"Model output: {model_output}")