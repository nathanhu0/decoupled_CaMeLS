#%%
from camels import CaMeLS
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import CACHE_DIR, set_seed, get_base_model, weighted_lm_loss, create_colored_text
import torch
import hydra
from hydra.utils import to_absolute_path
from doc_datasets import PairedNytDataset
import wandb 
import os
import higher 
import numpy as np

def adapt_base_model(base_model, weight_model, batch, inner_lr, outer_grad = True):
    optimizer = torch.optim.SGD(base_model.parameters(), lr=inner_lr)
    weights = weight_model(batch['adaptation_toks'], batch['adaptation_attn_mask'])
    
    with higher.innerloop_ctx(base_model, optimizer, copy_initial_weights=True, track_higher_grads = outer_grad) as (f_base_lm, diffopt):
        for i in range(len(batch['adaptation_toks'])):
            loss = weighted_lm_loss(f_base_lm, batch['adaptation_toks'][i:i+1], batch['adaptation_toks'][i:i+1], batch['adaptation_attn_mask'][i:i+1], weights[i:i+1])
            diffopt.step(loss)
    return f_base_lm, weights

def compute_outer_loss(base_model, weight_model, batch, inner_lr, outer_grad = True):
    paired_labels = batch['paired_toks'].clone()
    paired_labels[batch['paired_attn_mask']!=1] = -100
    #adaptation
    adapted_base_model, weights = adapt_base_model(base_model, weight_model, batch, inner_lr, outer_grad = outer_grad)
    if outer_grad:
        outer_loss = adapted_base_model(input_ids = batch['paired_toks'], attention_mask = batch['paired_attn_mask'], labels = paired_labels)['loss']
    else:
        with torch.no_grad():
            outer_loss = adapted_base_model(input_ids = batch['paired_toks'], attention_mask = batch['paired_attn_mask'], labels = paired_labels)['loss']
    return outer_loss, adapted_base_model, weights
    
def validate(base_model, weight_model, val_dataloader, inner_lr, base_state_dict, reset_base_frequency):
    losses = []
    weights = []
    base_model.load_state_dict(base_state_dict)
    for i, batch in enumerate(val_dataloader):
        batch = {k: v.to(base_model.device) for k, v in batch.items()}
        outer_loss, adapted_base_model, weights = compute_outer_loss(base_model, weight_model, batch, inner_lr, outer_grad = False)
        losses.append(outer_loss.item())
        weights.append((weights[batch['adaptation_attn_mask']!=0]).cpu().numpy())
        if (i+1) % reset_base_frequency == 0:
            base_model.load_state_dict(base_state_dict)
        else:
            base_model.load_state_dict(adapted_base_model.state_dict())
    cat_weights = np.concatenate(weights)
    return {'loss': sum(losses)/len(losses), 'mean_weight': cat_weights.mean(), 'std_weight': cat_weights.std(), 'max_weight': cat_weights.max(), 'min_weight': cat_weights.min()}

def plot_sample_weights(weight_model, batch, tokenizer, save_dir = None, wandb = False):
    with torch.no_grad():
        weights = weight_model(batch['adaptation_toks'], batch['adaptation_attn_mask'])
    sample_weights = []
    for i in range(len(batch['adaptation_toks'])):
        text = [tokenizer.decode(t) for t in batch['adaptation_toks'][i]][:sum(batch['adaptation_attn_mask'][i])]
        w = weights[i].detach().cpu().numpy()[:sum(batch['adaptation_attn_mask'][i])]
        sample_weights.append(create_colored_text(text, w))
    if save_dir is not None:
        for i, image in enumerate(sample_weights):
            image.save(os.path.join(save_dir, f'weights_{i}.png'))
    if wandb:
        wandb.log({'sample_weights': [wandb.Image(image) for image in sample_weights]})
#%%
@hydra.main(config_path='configs/metatrain', config_name='config')
def run(args):
    set_seed(args.seed)
    wandb.init(project='un-camels', entity='nathu', job_type='metatrain', config=args, name=args.notes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Loading models...')
    base_model = get_base_model(args.base_model, args.base_model_state_dict).to(device)
    base_state_dict = {k:v.detach().clone().cpu() for k, v in base_model.state_dict().items()}
    
    weight_model = CaMeLS(args.weight_model_base, args.weight_model_freeze_base).to(device)
    optimizer = torch.optim.Adam(list(weight_model.parameters()), lr=args.outer_lr)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir = CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    
    print('Loading dataset...')
    dataset = PairedNytDataset(to_absolute_path(args.data_path), tokenizer, args.max_length)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - args.val_size, args.val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last = True)
    train_iter = iter(train_dataloader)
    sample_weight_batch = next(iter(val_dataloader))
    sample_weight_batch = {k: v.to(device) for k, v in sample_weight_batch.items()}
    step = 0
    epoch = 0
    best_val_loss = float('inf')
    accumulated_losses = []
    
    print('Training...')
    while epoch < args.num_epochs:
        if step % 100 == 0:
            print(f'Epoch {epoch}, Step {step}')
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            epoch += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        outer_loss, adapted_base_model, _ = compute_outer_loss(base_model, weight_model, batch, args.inner_lr)
        accumulated_losses.append(outer_loss.item())
        outer_loss = outer_loss / args.gradient_accumulation_steps
        outer_loss.backward()
        #step the weight_model every gradient_accumulation_steps batches
        if (step+1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(weight_model.parameters(), args.gradiend_clip_threshold)  
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({'train_loss': sum(accumulated_losses), 'grad_norm': grad_norm})
            accumulated_losses = []
            
        #we either reset the base_model or load the post adaptation state_dict
        if (step+1) % args.reset_base_frequency == 0:
            base_model.load_state_dict(base_state_dict)
        else:
            base_model.load_state_dict(adapted_base_model.state_dict())
        del adapted_base_model
        
        #validation
        if (step+1) % (args.validation_frequency*args.grad_accumulation_steps) == 0:
            weight_model.eval()
            val_stats = validate(base_model, weight_model, val_dataloader, args.inner_lr, base_state_dict, args.reset_base_frequency)
            weight_model.train()
            val_loss = val_stats['loss']
            wandb.log({'val': val_stats})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(weight_model.state_dict(), f'{wandb.run.dir}/best_model.pt')
                print('Saved model')
            
            plot_sample_weights(weight_model, sample_weight_batch, tokenizer, wandb = True)
        step += 1
if __name__ == '__main__':
    run()
    
#%%
from hydra import compose, initialize
from omegaconf import OmegaConf

initialize(config_path='configs/metatrain')
args = compose(config_name="config")
print(OmegaConf.to_yaml(args))
# %%
