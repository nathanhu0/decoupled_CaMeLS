#%%
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from util import CACHE_DIR

class NytDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.df = pd.read_json(data_path)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir = CACHE_DIR)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        tokenized = self.tokenizer(text, padding = 'max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        return tokenized

class PairedTextDataset(Dataset):
    def __init__(self, tokenizer, max_length=1024):
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir = CACHE_DIR)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length
    def get_adaptation_passage(self, idx):
        raise NotImplementedError
    def get_paired_passage(self, idx):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, idx):
        adaptation_passage = self.get_adaptation_passage(idx)
        paired_passage = self.get_paired_passage(idx)
        adaptation_tokenized = self.tokenizer(adaptation_passage, padding = 'max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        paired_tokenized = self.tokenizer(paired_passage, padding = 'max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        return {'adaptation_toks': adaptation_tokenized['input_ids'].squeeze(), 
                'adaptation_attn_mask': adaptation_tokenized['attention_mask'].squeeze(), 
                'paired_toks': paired_tokenized['input_ids'].squeeze(), 
                'paired_attn_mask': paired_tokenized['attention_mask'].squeeze()}
    
class PairedNytDataset(PairedTextDataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        if '.csv' in data_path:
            self.df = pd.read_csv(data_path)
        elif '.json' in data_path:
            self.df = pd.read_json(data_path)
        super().__init__(tokenizer, max_length)
        
    def get_adaptation_passage(self, idx):
        return self.df.iloc[idx]['text']
    def get_paired_passage(self, idx):
        return self.df.iloc[idx]['match_text']
    def __len__(self):
        return len(self.df)
# %%
# tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir = CACHE_DIR)
# tokenizer.pad_token_id = tokenizer.eos_token_id
# train_dataset = NYT_dataset('/iris/u/nathu/un_camels/data/nyt/pretrain_split.csv', tokenizer, 1024)
# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# next(iter(train_dataloader))
# %%
