#%%
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Model
from util import CACHE_DIR
import torch

class CaMeLS(nn.Module):
    def __init__(self, pretrained_model='distilgpt2', freeze_base = False) -> None:
        super().__init__()
        self.pretrained_base = GPT2Model.from_pretrained(pretrained_model, cache_dir = CACHE_DIR)
        if freeze_base: 
            print('Freezing base model')
            for param in self.pretrained_base.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(self.pretrained_base.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.nl = nn.Softplus()
        with torch.no_grad():
            self.fc2.weight.fill_(0)
            self.fc2.bias.fill_(1)
            
    def forward(self, input_ids, attention_mask = None):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device = input_ids.device)
        base_outputs = self.pretrained_base(input_ids, attention_mask = attention_mask)['last_hidden_state']
        weights = self.nl(self.fc2(self.nl(self.fc1(base_outputs)))).squeeze()
        weights = weights*attention_mask
        return weights
# %%
