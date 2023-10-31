import torch
import torch.nn as nn
from torch.nn import Parameter
from higher.patch import monkeypatch as make_functional
import transformers
import sys
sys.path.append('../')
from util import CACHE_DIR
import re

class LoraModel(nn.Module):
    def __init__(self, model, r, param_re = "transformer.h.(\d+).attn.(c_attn|c_proj).weight"):
        super().__init__()

        self.base_model = model
        self.param_re = param_re
        self.lora_params = []
        self.lora_names = []
        for name, p in self.base_model.named_parameters():
            if p.dim() == 2 and re.match(self.param_re, name):
                A = Parameter(torch.zeros(p.shape[0], r))
                B = Parameter(torch.randn(r, p.shape[1]))
                self.register_parameter(name.replace('.', '_') + '_A', A)
                self.register_parameter(name.replace('.', '_') + '_B', B)
                self.lora_params.append((A, B))
                self.lora_names.append(name)
            else:
                self.lora_params.append(None)
        print(f'Initialized LORA parameters for {len(self.lora_names)} modules', self.lora_names)
        #self.lora_params = nn.ParameterList([p for p in self.lora_params if p is not None for p in p])

    def forward(self, LORA=True, *args, **kwargs):
        if LORA:
            fmodel = make_functional(self.base_model)
            new_params = [p if lp is None else p + lp[0] @ lp[1] for p, lp in zip(self.base_model.parameters(), self.lora_params)]
            return fmodel(*args, **kwargs, params=new_params)
        else:
            return self.base_model(*args, **kwargs)

    def lora_parameters(self):
        return nn.ParameterList([p for lp in self.lora_params if lp is not None for p in lp])
    
    def save_lora(self, path):
        torch.save(self.lora_params, path)
        
    def load_lora(self, path):
        loaded_lora_params = torch.load(path)
        for original, loaded in zip(self.lora_params, loaded_lora_params):
            if original is None:
                if loaded is not None:
                    print('Warning: loaded LORA parameters but model does not have LORA parameters')
            else:
                if loaded is None:
                    print('Warning: model has LORA parameters but loaded LORA parameters is None')
                else:
                    original[0].data = loaded[0].data
                    original[1].data = loaded[1].data
        