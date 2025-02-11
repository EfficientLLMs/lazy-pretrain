

MODEL_MAP = {
    'OLMo-1B': 'allenai/OLMo-1B',
    'OLMo-7B': 'allenai/OLMo-7B',
}

class OLMo_1B:
    layers = 16
    hidden_size = 2048
    attn_head = 16
    context_length = 2048

class OLMo_7B:
    layers = 32
    hidden_size = 4096
    attn_head = 32
    context_length = 4096