#%%
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Model
from util import CACHE_DIR, create_colored_text
import wandb
import os
import torch


class CaMeLS(nn.Module):
    def __init__(self, pretrained_model='distilgpt2', freeze_base = False, nl = 'softplus') -> None:
        super().__init__()
        self.pretrained_base = GPT2Model.from_pretrained(pretrained_model, cache_dir = CACHE_DIR)
        if freeze_base: 
            print('Freezing base model')
            for param in self.pretrained_base.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(self.pretrained_base.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        if nl == 'softplus':
            self.nl = nn.Softplus()
            with torch.no_grad():
                self.fc2.weight.fill_(0)
                self.fc2.bias.fill_(1)
        elif nl == 'sigmoid':
            self.nl = nn.Sigmoid()
            with torch.no_grad():
                self.fc2.weight.fill_(0)
                self.fc2.bias.fill_(0)
        else:
            raise ValueError(f'Non-implemented nonlinearity {nl}')
        
            
    def forward(self, input_ids, attention_mask = None):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device = input_ids.device)
        base_outputs = self.pretrained_base(input_ids, attention_mask = attention_mask)['last_hidden_state']
        weights = self.nl(self.fc2(self.nl(self.fc1(base_outputs)))).squeeze()
        weights = weights*attention_mask
        return weights
# %%
def plot_sample_weights(weight_model, batch, tokenizer, save_dir = None, log_to_wandb = False):
    with torch.no_grad():
        weights = weight_model(batch['adaptation_toks'], batch['adaptation_attn_mask'])
    sample_weights = []
    for i in range(len(batch['adaptation_toks'])):
        text = [tokenizer.decode(t) for t in batch['adaptation_toks'][i]][:sum(batch['adaptation_attn_mask'][i])]
        w = weights[i].detach().cpu().numpy()[:sum(batch['adaptation_attn_mask'][i])]
        sample_weights.append(create_colored_text(text, w))
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok = True)
        for i, image in enumerate(sample_weights):
            image.save(os.path.join(save_dir, f'weights_{i}.png'))
    if log_to_wandb:
        for i, image in enumerate(sample_weights):
            wandb.log({f'sample_weights_{i}': wandb.Image(image)})