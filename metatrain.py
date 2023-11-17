#%%
from camels import CaMeLS, plot_sample_weights
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import CACHE_DIR, set_seed, get_base_model, weighted_lm_loss, debug_memory
import torch
import hydra
from hydra.utils import to_absolute_path
from doc_datasets import PairedNytDataset
import wandb 
import os
import higher 
import numpy as np

def adapt_base_model(base_model, weight_model, batch, inner_lr, outer_grad = True, debug = False):
    if debug: 
        print(' Pre inner loop: ')
        debug_memory()
    optimizer = torch.optim.SGD(base_model.parameters(), lr=inner_lr)
    weights = weight_model(batch['adaptation_toks'], batch['adaptation_attn_mask'])
    if debug: 
        print(' Post Weight model fwd pass: ')
        debug_memory()
    with higher.innerloop_ctx(base_model, optimizer, copy_initial_weights=True, track_higher_grads = outer_grad) as (f_base_lm, diffopt):
        for i in range(len(batch['adaptation_toks'])):
            
            loss = weighted_lm_loss(f_base_lm, batch['adaptation_toks'][i:i+1], batch['adaptation_toks'][i:i+1], batch['adaptation_attn_mask'][i:i+1], weights[i:i+1])
            if debug: 
                print(f' {i}th inner loss compute: ')
                debug_memory()
            diffopt.step(loss)
            if debug: 
                print(f' {i}th inner loss backwards: ')
                debug_memory()
            
    return f_base_lm, weights

def compute_outer_loss(base_model, weight_model, batch, inner_lr, outer_grad = True, debug=False):
    #adaptation
    adapted_base_model, weights = adapt_base_model(base_model, weight_model, batch, inner_lr, outer_grad = outer_grad)
    if debug: 
        print(' Post inner loop: ')
        debug_memory()
    with torch.set_grad_enabled(outer_grad):
        paired_outer_loss = adapted_base_model(input_ids = batch['paired_toks'], attention_mask = batch['paired_attn_mask'], labels = batch['paired_labels'])['loss']
        future_outer_loss = adapted_base_model(input_ids = batch['future_toks'], attention_mask = batch['future_attn_mask'], labels = batch['future_labels'])['loss']
    if debug: 
        print(' Post Outer Loss: ')
        debug_memory()
    return paired_outer_loss, future_outer_loss, adapted_base_model, weights
    
def validate(base_model, weight_model, val_dataloader, inner_lr, base_state_dict, reset_base_frequency, debug = False):
    paired_losses = []
    future_losses = []
    weight_list = []
    base_model.load_state_dict(base_state_dict)
    for i, batch in enumerate(val_dataloader):
        if debug and i > 50:
            break
        batch = {k: v.to(base_model.device) for k, v in batch.items()}
        paired_outer_loss, future_outer_loss, adapted_base_model, weights = compute_outer_loss(base_model, weight_model, batch, inner_lr, outer_grad = False)
        paired_losses.append(paired_outer_loss.item())
        future_losses.append(future_outer_loss.item())
        weight_list.append((weights[batch['adaptation_attn_mask']!=0]).detach().cpu().numpy())
        if (i+1) % reset_base_frequency == 0:
            base_model.load_state_dict(base_state_dict)
        else:
            base_model.load_state_dict(adapted_base_model.state_dict())
    cat_weights = np.concatenate(weight_list)
    return {'paired_loss': sum(paired_losses)/len(paired_losses), 
            'future_loss': sum(future_losses)/len(future_losses),
            'mean_weight': cat_weights.mean(), 
            'std_weight': cat_weights.std(), 
            'max_weight': cat_weights.max(), 
            'min_weight': cat_weights.min()}


#%%
@hydra.main(config_path='configs', config_name='metatrain_config')
def run(args):
    set_seed(args.seed)
    wandb.init(project='un-camels', entity='nathu', job_type='metatrain', config=args, name=args.notes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Loading models...')
    base_model = get_base_model(args.base_model, args.base_model_state_dict).to(device)
    base_state_dict = {k:v.detach().clone().cpu() for k, v in base_model.state_dict().items()}
    
    weight_model = CaMeLS(args.weight_model_base, args.weight_model_freeze_base, nl=args.weight_model_nl).to(device)
    optimizer = torch.optim.Adam(list(weight_model.parameters()), lr=args.outer_lr)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir = CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    
    print('Loading dataset...')
    
    if args.dataset.name == 'nyt':
        train_dataset = PairedNytDataset(to_absolute_path(args.dataset.train_data_path), tokenizer, args.dataset.max_length)
        val_dataset = PairedNytDataset(to_absolute_path(args.dataset.val_data_path), tokenizer, args.dataset.max_length)
    else:
        raise NotImplementedError
       
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last = True)
    train_iter = iter(train_dataloader)
    sample_weight_batch = next(iter(val_dataloader))
    sample_weight_batch = {k: v.to(device) for k, v in sample_weight_batch.items()}
    step = 0
    epoch = 0
    best_val_loss = float('inf')
    
    accumulated_total_losses = []
    accumulated_paired_losses = []
    accumulated_future_losses = []
    
    print('Training...')
    while epoch < args.num_epochs:
        if step % 100 == 0:
            print(f'Epoch {epoch}, Step {step}')
            
        if args.debug: 
            print(' --- Before step: ', step, '----')
            debug_memory() 
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            epoch += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        paired_outer_loss, future_outer_loss, adapted_base_model, _ = compute_outer_loss(base_model, weight_model, batch, args.inner_lr, debug = args.debug)
        
        total_outer_loss = paired_outer_loss*(1 - args.future_loss_weight) + future_outer_loss*args.future_loss_weight
        
        if args.debug: 
            print(' Post outer loss computation: ', step)
            debug_memory() 
        
        accumulated_total_losses.append(total_outer_loss.item())
        accumulated_paired_losses.append(paired_outer_loss.item())
        accumulated_future_losses.append(future_outer_loss.item())
        
        total_outer_loss = total_outer_loss / args.gradient_accumulation_steps
        total_outer_loss.backward()
        
        if args.debug: 
            print(' Post outer loss backwards: ', step)
            debug_memory() 
        
        #step the weight_model every gradient_accumulation_steps batches
        if (step+1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(weight_model.parameters(), args.gradient_clip_threshold)  
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({ 'train_loss': np.mean(accumulated_total_losses),
                        'paired_loss': np.mean(accumulated_paired_losses),
                        'future_loss': np.mean(accumulated_future_losses),
                        'grad_norm': grad_norm})
            
            accumulated_total_losses = []
            accumulated_paired_losses = []
            accumulated_future_losses = []
            
        if args.debug: 
            print(' Post Optimizer Step: ')
            debug_memory()
        #we either reset the base_model or load the post adaptation state_dict
        if (step+1) % args.reset_base_frequency == 0:
            base_model.load_state_dict(base_state_dict)
        else:
            base_model.load_state_dict(adapted_base_model.state_dict())
        del adapted_base_model
        if args.debug: 
            print(' Base Model State Update: ')
            debug_memory()
        #validation
        if (step+1) % (args.validation_frequency*args.gradient_accumulation_steps) == 0:
            weight_model.eval()
            val_stats = validate(base_model, weight_model, val_dataloader, args.inner_lr, base_state_dict, args.reset_base_frequency, debug = args.debug)
            weight_model.train()
            val_stats['total_loss'] = val_stats['paired_loss']*(1 - args.future_loss_weight) + val_stats['future_loss']*args.future_loss_weight
            val_loss = val_stats['total_loss']
            wandb.log({'val': val_stats})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(weight_model.state_dict(), f'./best_model.pt')
                print('Saved model')
            
            plot_sample_weights(weight_model, sample_weight_batch, tokenizer, log_to_wandb = True)
        step += 1
if __name__ == '__main__':
    run()
    

# a useful snippet for debugging
#%%
# from hydra import compose, initialize
# from omegaconf import OmegaConf

# initialize(config_path='configs/metatrain')
# args = compose(config_name="config")
# print(OmegaConf.to_yaml(args))
# %%
