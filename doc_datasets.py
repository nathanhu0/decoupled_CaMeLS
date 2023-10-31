#%%
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from util import CACHE_DIR, shuffle_groups, return_k_unique
import torch
import copy


def tokenize_qa(question, answer, tokenizer, max_length=1024):
    #assume question and answer are text strings but fully processed (eos, spaced, captialized, etc)
    tok_answer = tokenizer(answer, return_tensors="pt")
    tok_question = tokenizer(question, return_tensors="pt")
    qa_ids = torch.cat([tok_question['input_ids'], (tok_answer['input_ids'])], 1) #[1, len_q + len_a]
    if qa_ids.shape[1] > max_length:
        print(f'total question len {qa_ids.shape[1]} excedes max_question len f{max_length}. Truncating:')
        print(question, answer)
        num_to_truncate = qa_ids.shape[1] - max_length
        qa_ids = qa_ids[:, :-num_to_truncate]
        tok_question['input_ids'] = tok_question['input_ids'][:, :-num_to_truncate]
        tok_question['attention_mask'] = tok_question['attention_mask'][:, :-num_to_truncate]
    #TODO FINISH
    n_pad = max_length - qa_ids.shape[1]
    qa_attention = torch.cat([tok_question['attention_mask'], (tok_answer['attention_mask'])], 1)
    qa_target_ids = qa_ids.clone()
    qa_target_ids[:, :tok_question['input_ids'].shape[1]] = -100
    
    qa_ids = torch.nn.functional.pad(qa_ids, (0, n_pad), value = tokenizer.pad_token_id)
    qa_attention = torch.nn.functional.pad(qa_attention, (0, n_pad), value = 0)
    qa_target_ids = torch.nn.functional.pad(qa_target_ids, (0, n_pad), value = -100)
    return qa_ids, qa_attention, qa_target_ids
class NytDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        if '.csv' in data_path:
            self.df = pd.read_csv(data_path)
        elif '.json' in data_path:
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
        self.text_fn_dic = {}
   
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, idx):
        
        return_dic = {}
        for text_name, retr_fn in self.text_fn_dic.items():
            text = retr_fn(idx)
            if isinstance(text, str):
                tokenized = self.tokenizer(text, padding = 'max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
                return_dic[text_name + '_toks'] = tokenized['input_ids'].squeeze()
                return_dic[text_name + '_attn_mask'] = tokenized['attention_mask'].squeeze()
                return_dic[text_name + '_labels'] = tokenized['input_ids'].squeeze().clone()
                return_dic[text_name + '_labels'][return_dic[text_name + '_attn_mask'] == 0] = -100
            else: #in the case that text is a tuple, we assume it is a question answer pair
                question, answer = text
                qa_ids, qa_attention, qa_target_ids = tokenize_qa(question, answer, self.tokenizer, self.max_length)
                return_dic[text_name + '_toks'] = qa_ids.squeeze()
                return_dic[text_name + '_attn_mask'] = qa_attention.squeeze()
                return_dic[text_name + '_labels'] = qa_target_ids.squeeze()
                
        return return_dic
     
class PairedNytDataset(PairedTextDataset):
    def __init__(self, data_path, tokenizer, max_length=1024, ):
        if '.csv' in data_path:
            self.df = pd.read_csv(data_path)
        elif '.json' in data_path:
            self.df = pd.read_json(data_path)
        self.df.dropna(inplace=True, subset=['match_text'])
        self.df['date'] = pd.to_datetime(self.df['date'])
        super().__init__(tokenizer, max_length)
        self.text_fn_dic = {'adaptation': self.get_adaptation_passage, 
                            'paired': self.get_paired_passage, 
                            'future': self.get_future_passage}
    def get_adaptation_passage(self, idx):
        return self.df.iloc[idx]['text']
    def get_paired_passage(self, idx):
        return self.df.iloc[idx]['match_text']
    def get_future_passage(self, idx):
        date = self.df.iloc[idx]['date']
        next_date = self.df[self.df['date'] > date]['date'].min()
        next_df = self.df[self.df['date'] == next_date]
        if len(next_df) == 0:
            return self.df[self.df['date'] == date]['text'].sample().item()
        else:
            return next_df['text'].sample().item()
        
    def __len__(self):
        return len(self.df)
    
class ArchivalQADataset(PairedTextDataset):
    def __init__(self, csv_path, full_passage = False, shuffle_by='doc_id', downsample_to=-1,downsample_by='ans_paragraph', **kwargs):
        self.csv_path = csv_path
        self.full_passage = full_passage
        self.data_frame = pd.read_csv(csv_path)
        #we sort pre shuffle to make sure that for any given doc_id, the examples are in increasing order of para_num
        self.data_frame.sort_values('para_num', kind='stable', inplace=True)
        self.data_frame = shuffle_groups(self.data_frame, shuffle_by)
        if downsample_to > 0:
            self.data_frame = return_k_unique(self.data_frame, downsample_to, downsample_by)
        super().__init__(**kwargs)
        self.text_fn_dic = {'adaptation': self.get_text,
                            'qa': self.get_qa}
        
    def __len__(self):
        return len(self.data_frame)
    
    def get_qa(self, idx):
        row = self.data_frame.iloc[idx]
        answer = row['answer']
        if answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        question = row['question'].strip() 
        answer = ' ' + answer.strip() + self.tokenizer.eos_token
        return question, answer
    
    def get_text(self, idx):
        if self.full_passage:
            return self.data_frame.iloc[idx]['ans_text']
        return self.data_frame.iloc[idx]['ans_paragraph']

    def get_deduplicated_dataset(self):
        new_arch_ds = copy.deepcopy(self)
        if self.full_passage:
            new_arch_ds.data_frame = self.data_frame.drop_duplicates(subset=['ans_text'])
        else:
            new_arch_ds.data_frame = self.data_frame.drop_duplicates(subset=['ans_paragraph'])
        return new_arch_ds
    
    def __getitem__(self, idx):
        return_dic = super().__getitem__(idx)
        #add qa
        return return_dic
        
        
# %%
# tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir = CACHE_DIR)
# tokenizer.pad_token_id = tokenizer.eos_token_id
# train_dataset = NYT_dataset('/iris/u/nathu/un_camels/data/nyt/pretrain_split.csv', tokenizer, 1024)
# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# next(iter(train_dataloader))
# %%
