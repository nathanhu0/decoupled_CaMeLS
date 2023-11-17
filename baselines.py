#various classes for baseline models all implement a forward method that takes input_ids and attention_mask as arguments, the same inputs as CaMeLS


#%%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy
from util import CACHE_DIR
import numpy as np

def get_nes_from_toks(toks, tokenizer, nlp = None, entities_to_ignore = []):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    text_by_toks = [tokenizer.decode(t, clean_up_tokenization_spaces=False) for t in toks]
    text= ''.join(text_by_toks)
    is_ne = [0 for _ in range(len(text))]
    doc = nlp(text)
    cur_idx = 0
    for token in doc:
        start_idx = text.find(token.text, cur_idx)
        for j in range(len(token)):
            if token.ent_type_ and token.ent_type_ not in entities_to_ignore:
                is_ne[start_idx + j] = 1
            else:
                is_ne[start_idx + j] = 0 
        cur_idx = start_idx + len(token)
    tok_lens = [len(t) for t in text_by_toks]
    prefix_sum = np.cumsum(tok_lens, dtype=np.int32)
    return  [max(is_ne[prefix_len-tok_len:prefix_len]) for tok_len,prefix_len in zip(tok_lens, prefix_sum)] 
 
class SalientSpanWeighting(nn.Module):
    def __init__(self, tokenizer='gpt2', entities_to_ignore=['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']):
        self.nlp = spacy.load("en_core_web_sm")
        super().__init__()
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir = CACHE_DIR)
        else:   
            self.tokenizer = tokenizer
        self.entities_to_ignore = entities_to_ignore            
    def forward(self, input_ids, attention_mask = None):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device = input_ids.device)
        batch_weights = []
        for i in range(len(input_ids)):
            named_ents = get_nes_from_toks(input_ids[i][attention_mask[i]==1], self.tokenizer, self.nlp, entities_to_ignore=self.entities_to_ignore)
            padding = [0]*(len(input_ids[i]) - len(named_ents))
            weights = np.concatenate((named_ents, padding))
            batch_weights.append(weights)
        
        return torch.tensor(np.stack(batch_weights)).to(input_ids.device)

#%%
class UniformWeighting(nn.Module):
    def forward(self, input_ids, attention_mask = None):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device = input_ids.device)
        return attention_mask
# %%
