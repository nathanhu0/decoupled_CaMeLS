
#%%
import hydra
import wandb
from tqdm import tqdm
from util import CACHE_DIR, get_base_model
from transformers import AutoTokenizer
from doc_datasets import NytDataset
import torch
from hydra.utils import to_absolute_path



def validate_model(model, val_dataloader, args):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            losses.append(loss.item())
    return sum(losses)/len(losses)
#%%
@hydra.main(config_path='configs/pretraining', config_name='config')
def run(args):
    model = get_base_model(args.base_model, args.base_model_state_dict).to(args.device)
    if args.gradient_checkpointing:
        model.transformer.gradient_checkpointing=True
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir = CACHE_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset = NytDataset(to_absolute_path(args.train_data_path), tokenizer, args.max_length)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = NytDataset(to_absolute_path(args.val_data_path), tokenizer, args.max_length)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    wandb.init(project='un-camels', entity='nathu', job_type='pretraining', config=args)
    val_time_steps = [int(len(train_dataloader)*i/args.validations_per_epoch)-1 for i in range(args.validations_per_epoch)]
    
    best_val_loss = float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch}')
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss/args.gradient_accumulation_steps
            loss.backward()
            wandb.log({'train_loss': loss.item()})
            if (i+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
            if i in val_time_steps:
                val_loss = validate_model(model, val_dataloader, args)
                wandb.log({'val_loss': val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{wandb.run.dir}/best_model.pt')
                    print('Saved model')
                model.train()
    
if __name__ == '__main__':
    run()
# %%