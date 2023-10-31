#%%
import hydra 
import wandb
from tqdm import tqdm
from util import CACHE_DIR, get_base_model
from transformers import AutoTokenizer
from doc_datasets import ArchivalQADataset
import torch
from hydra.utils import to_absolute_path
import numpy as np
from lora import LoraModel

#%%

def validate_model(lora_model, val_dataloader, args):
    print('validating...')
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = lora_model(input_ids = batch['qa_toks'], attention_mask = batch['qa_attn_mask'], labels = batch['qa_labels'], LORA = True)
            loss = outputs.loss
            losses.append(loss.item())
    return sum(losses)/len(losses)


@hydra.main(config_path='configs', config_name='pretrain_lora_config')
def run(args):
    print('loading model...')
    model = get_base_model(args.base_model, args.base_model_state_dict).to(args.device)
    lora_model = LoraModel(model, args.r, args.param_re).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir = CACHE_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print('loading dataset...')
    if args.dataset.name == 'archivalqa':
        train_dataset = ArchivalQADataset(to_absolute_path(args.dataset.qa_pretrain_data_path), tokenizer = tokenizer, max_length = args.dataset.max_length)
        val_dataset = ArchivalQADataset(to_absolute_path(args.dataset.qa_pretrain_val_data_path), tokenizer = tokenizer, max_length = args.dataset.max_length)
    else:
        raise ValueError(f'Invalid dataset {args.dataset.name}')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    wandb.init(project='un-camels', entity='nathu', job_type='lora_pretraining', config=args)
    
    val_time_steps = [int(len(train_dataloader)*i/args.validations_per_epoch)-1 for i in range(args.validations_per_epoch)]
    
    best_val_loss = float('inf')
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=args.lr)
    elif args.optimizer == 'adafactor':
        optimizer = torch.optim.Adafactor(lora_model.lora_parameters(), lr=args.lr)
    else:
        raise ValueError(f'Invalid optimizer {args.optimizer}')
    
    losses = []
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch}')
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = lora_model(input_ids = batch['qa_toks'], attention_mask = batch['qa_attn_mask'], labels = batch['qa_labels'], LORA = True)
            loss = outputs.loss
            losses.append(loss.item())
            loss = outputs.loss/args.gradient_accumulation_steps
            loss.backward()
        
            if (i+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    base_model_hash = np.mean([p.mean().item() for p in lora_model.base_model.parameters()])
                
                wandb.log({'train_loss': np.mean(losses), 
                        'base_model_hash': base_model_hash})
                
                losses = [] 
                
            if i in val_time_steps:
                val_loss = validate_model(lora_model, val_dataloader, args)
                wandb.log({'val_loss': val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    lora_model.save_lora(f'./best_lora.pt')
                    print('Saved model')
                lora_model.train()
            
if __name__ == '__main__':
    run()
#
#%% a useful snippet for debugging
# from hydra import compose, initialize
# from omegaconf import OmegaConf

# initialize(config_path='configs')
# args = compose(config_name="pretrain_lora_config")
# print(OmegaConf.to_yaml(args))
# %%