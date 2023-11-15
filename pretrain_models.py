
#simple training loop to pretain LMs on text distribution
# %%
import hydra
import wandb
from tqdm import tqdm
from util import CACHE_DIR, get_base_model
from transformers import AutoTokenizer
from doc_datasets import NytDataset, ArchivalQADataset
import torch
from hydra.utils import to_absolute_path
import numpy as np

def validate_model(model, val_dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['adaptation_toks'], attention_mask=batch['adaptation_attn_mask'], labels=batch['adaptation_labels'])
            loss = outputs.loss
            losses.append(loss.item())
    return sum(losses)/len(losses)
#%%
@hydra.main(config_path='configs', config_name='pretraining_config')
def run(args):
    print('loading model...')
    model = get_base_model(args.base_model, args.base_model_state_dict).to(args.device)
    if args.gradient_checkpointing:
        model.transformer.gradient_checkpointing=True
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir = CACHE_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print('loading dataset...')
    if args.dataset.name == 'nyt':
        train_dataset = NytDataset(to_absolute_path(args.dataset.pretrain_data_path), tokenizer, args.dataset.max_length)
        val_dataset = NytDataset(to_absolute_path(args.dataset.pretrain_val_data_path), tokenizer, args.dataset.max_length)

    elif args.dataset.name == 'archivalqa':
        train_dataset = ArchivalQADataset(to_absolute_path(args.dataset.qa_pretrain_data_path), tokenizer=tokenizer, full_passage=args.dataset.full_passage, max_length=args.dataset.max_length)
        val_dataset = ArchivalQADataset(to_absolute_path(args.dataset.qa_pretrain_val_data_path), tokenizer=tokenizer, full_passage=args.dataset.full_passage, max_length=args.dataset.max_length)
        
    else:
        raise ValueError(f'Invalid dataset {args.dataset.name}')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    wandb.init(project='un-camels', entity='nathu', job_type='pretraining', config=args)
    val_time_steps = [int(len(train_dataloader)*i/args.validations_per_epoch)-1 for i in range(args.validations_per_epoch)]
    
    best_val_loss = float('inf')
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adafactor':
        optimizer = torch.optim.Adafactor(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f'Invalid optimizer {args.optimizer}')
    
    losses = []
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch}')
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(input_ids=batch['adaptation_toks'], attention_mask=batch['adaptation_attn_mask'], labels=batch['adaptation_labels'])
            loss = outputs.loss
            losses.append(loss.item())
            loss = outputs.loss/args.gradient_accumulation_steps
            loss.backward()
            
            if (i+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({'train_loss': np.mean(losses)})
                losses = [] 
        
            if i in val_time_steps:
                val_loss = validate_model(model, val_dataloader, args.device)
                wandb.log({'val_loss': val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{hydra.run.dir}/best_model.pt')
                    print('Saved model')
                model.train()
    
if __name__ == '__main__':
    run()
# %%
