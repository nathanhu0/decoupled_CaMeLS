#%%
import hydra 
import wandb
from tqdm import tqdm
from util import CACHE_DIR, get_base_model
from transformers import AutoTokenizer, Adafactor
from doc_datasets import ArchivalQADataset
import torch
from hydra.utils import to_absolute_path
import numpy as np
from lora import add_lora, set_lora_state, get_lora_parameters, freeze_non_lora

#%%

def validate_model(model, val_dataloader, args):
    print('validating...')
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(input_ids = batch['qa_toks'], attention_mask = batch['qa_attn_mask'], labels = batch['qa_labels'])
            loss = outputs.loss
            losses.append(loss.item())
    return sum(losses)/len(losses)


@hydra.main(config_path='configs', config_name='pretrain_lora_config')
def run(args):
    print('loading model...')
    model = get_base_model(args.base_model, to_absolute_path(args.base_model_state_dict))
    model.eval()
    add_lora(model, args.r)
    freeze_non_lora(model, lora_gradient = True)
    model.to(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir = CACHE_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print('loading dataset...')
    if args.dataset.name == 'archivalqa':
        train_dataset = ArchivalQADataset(to_absolute_path(args.dataset.qa_pretrain_data_path), tokenizer=tokenizer, full_passage=args.dataset.full_passage, max_length=args.dataset.max_length)
        val_dataset = ArchivalQADataset(to_absolute_path(args.dataset.qa_pretrain_val_data_path), tokenizer=tokenizer, full_passage=args.dataset.full_passage, max_length=args.dataset.max_length)
    else:
        raise ValueError(f'Invalid dataset {args.dataset.name}')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    if args.debug:
        sample_batch = next(iter(train_dataloader))
        sample_batch = {k: v.to(args.device) for k, v in sample_batch.items()}
    wandb.init(project='un-camels', entity='nathu', job_type='lora_pretraining', config=args)
    
    val_time_steps = [int(len(train_dataloader)*i/args.validations_per_epoch) for i in range(args.validations_per_epoch)] 
    
    best_val_loss = float('inf')
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(get_lora_parameters(model), lr=args.lr)
    elif args.optimizer == 'adafactor':
        optimizer = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False)
    else:
        raise ValueError(f'Invalid optimizer {args.optimizer}')
    
    losses = []
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch}')
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(input_ids = batch['qa_toks'], attention_mask = batch['qa_attn_mask'], labels = batch['qa_labels'])
            loss = outputs.loss
            losses.append(loss.item())
            loss = outputs.loss/args.gradient_accumulation_steps
            loss.backward()
        
            if (i+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if args.debug:
                    with torch.no_grad():
                        set_lora_state(model, False)
                        base_model_loss = model(input_ids = sample_batch['qa_toks'], attention_mask = sample_batch['qa_attn_mask'], labels = sample_batch['qa_labels']).loss.item()
                        wandb.log({'base_model_sample_loss': base_model_loss}, commit=False)
                        set_lora_state(model, True)
                wandb.log({'train_loss': np.mean(losses)})
                
                losses = [] 
                
            if i in val_time_steps:
                val_loss = validate_model(model, val_dataloader, args)
                wandb.log({'val_loss': val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(get_lora_parameters(model), f'./best_lora.pt')
                    print('Saved model')
                
            
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