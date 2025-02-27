import torch
from hf_olmo import OLMoForCausalLM, OLMoConfig
import yaml
import os

model_file = "models/tiny-olmo-150M-step406934-unsharded/model.pt"
train_file = "models/tiny-olmo-150M-step406934-unsharded/train.pt"
config_file = "models/tiny-olmo-150M-step406934-unsharded/config.yaml"


class TinyOLMo(OLMoForCausalLM):

    def __init__(self, model_dir = None, model = None, init_params: bool = False):
        self.model_dir = model_dir

        if model_dir:
            self.model_file = os.path.join(model_dir, 'model.pt')
            self.train_file = os.path.join(model_dir, 'train.pt')
            self.config_file = os.path.join(model_dir, 'config.yaml')

        with open(config_file, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.SafeLoader)
        
        config = OLMoConfig(**config_data['model'])
        
        super(TinyOLMo, self).__init__(config, model, init_params)

        if not torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(model_file))
        

# Load state dict
tiny_olmo = TinyOLMo(model_dir='models/tiny-olmo-150M-step406934-unsharded')

print(tiny_olmo)