#%%
import torch
import torch.nn as nn
from torch.nn import Parameter
from higher.patch import monkeypatch as make_functional
import transformers
import sys
sys.path.append('../')
from util import CACHE_DIR
import re

#wrapper class for linear layer
class LoraLinear(nn.Module):
    def __init__(self, original_linear, r):
        super().__init__()
        self.A = Parameter(torch.zeros(original_linear.weight.shape[0], r)) 
        self.B = Parameter(torch.randn(r, original_linear.weight.shape[1]))
        self.original_linear = original_linear
        self.lora = True
    
    def forward(self, *args, **kwargs):
        orig_out = self.original_linear(*args, **kwargs)
        if self.lora:
            #print([x.shape for x in args], [k for k in kwargs])
            #print(self.original_linear.weight.shape, self.A.shape, self.B.shape, orig_out.shape)
            
            if isinstance(self.original_linear, transformers.models.gpt2.modeling_gpt2.Conv1D):
                return orig_out + (args[0] @ self.A) @ self.B #check shapes
            else:
                return orig_out + (args[0]@ self.B.transpose(-1,-2)) @ self.A.transpose(-1,-2) #check shapes
        else:
            return orig_out

#patch modules in place
def add_lora(model, r):
    for parent_module in list(model.modules()):
        for name, module in parent_module.named_children():
            if isinstance(module, nn.Linear) or isinstance(module, transformers.models.gpt2.modeling_gpt2.Conv1D):
                setattr(parent_module, name, LoraLinear(module, r))

#turn lora on or off via
def set_lora_state(model, lora):
    for module in list(model.modules()):
        if isinstance(module, LoraLinear):
            module.lora = lora

def get_lora_parameters(model):
    parameter_list = []
    for module in list(model.modules()):
        if isinstance(module, LoraLinear):
            parameter_list.append(module.A)
            parameter_list.append(module.B)
    return parameter_list
#set all non-lora params to not require gradient. Lora param's require_grad is set to lora_gradient 
def freeze_non_lora(model, lora_gradient = True):
    for param in model.parameters():
        param.requires_grad = False
    for param in get_lora_parameters(model):
        param.requires_grad = lora_gradient
  
    
    

#%%

# from transformers import AutoTokenizer, AutoModelForCausalLM
# from util import CACHE_DIR
# model=AutoModelForCausalLM.from_pretrained('distilgpt2', cache_dir=CACHE_DIR)
# add_lora(model, 2)
# sample_input = torch.tensor([5,21,500,100])
# out = model(sample_input)

#%%

#WE NO LONGER USE FUNCTIONAL IMPLEMENTATION OF LORA
class LoraModel(nn.Module):
    def __init__(self, model, r, param_re = "transformer.h.(\d+).attn.(c_attn|c_proj).weight"):
        super().__init__()

        self.base_model = model
        self.param_re = param_re
        self.lora_params = []
        self.lora_names = []
        self.lora = True
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
        
        self.functional = make_functional(self.base_model)
        #self.lora_params = nn.ParameterList([p for p in self.lora_params if p is not None for p in p])


    def adapt_parameters(self, model_parameters):
        """
        Adapt the parameters of the model to the LORA parameters.
        model_parameters: an iterable of parameters
        returns a list of parameters
        """
        new_params = [p if lp is None else p + lp[0] @ lp[1] for p, lp in zip(model_parameters, self.lora_params)]
        return new_params
        
    def forward(self, LORA=None, base_parameters = None, *args, **kwargs):
        """
        A forward pass through the model.
        LORA: whether to use LORA parameters, if None, use self.lora. When LORA is False, base_parameters is ignored and a forward pass is made through the base model.
        base_parameters: if LORA is True, the base parameters to use. If None, use self.base_model.parameters()
        """
        
        if LORA is None:
            LORA = self.lora
        
        if base_parameters is None:
            base_parameters = self.base_model.parameters()
        elif not LORA:
            print('Warning: LORA is False but base_parameters is not None. Ignoring base_parameters.')
        if LORA:
            new_params = self.adapt_parameters(base_parameters)
            return self.functional(*args, **kwargs, params=new_params)
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
        
# %%
