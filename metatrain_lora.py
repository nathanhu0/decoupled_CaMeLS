#%%
from camels import CaMeLS, plot_sample_weights
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import CACHE_DIR, set_seed, get_base_model, weighted_lm_loss, create_colored_text, debug_memory
import torch
import hydra
from hydra.utils import to_absolute_path
from doc_datasets import PairedNytDataset, ArchivalQADataset
import wandb 
import os
import higher 
import numpy as np
from lora import add_lora, set_lora_state, load_lora, get_lora_parameters

def adapt_base_model(base_model, weight_model, batch, inner_lr, outer_grad = True, debug = False, inner_lora = False):
    #inner_lora: if true, we use the lora model to compute the inner loss. If false, we use the base model
    
    if debug: 
        print(' Pre inner loop: ')
        debug_memory()
    optimizer = torch.optim.SGD(base_model.parameters(), lr=inner_lr)
    weights = weight_model(batch['adaptation_toks'], batch['adaptation_attn_mask'])
    if debug: 
        print(' Post Weight model fwd pass: ')
        debug_memory()
    
    set_lora_state(base_model, inner_lora)
    
    inner_losses = []
    with higher.innerloop_ctx(base_model, optimizer, copy_initial_weights=True, track_higher_grads = outer_grad) as (f_base_lm, diffopt):
        for i in range(len(batch['adaptation_toks'])):
            
            loss = weighted_lm_loss(f_base_lm, batch['adaptation_toks'][i:i+1], batch['adaptation_toks'][i:i+1], batch['adaptation_attn_mask'][i:i+1], weights[i:i+1])
            
            inner_losses.append(loss.item())
            if debug: 
                print(f' {i}th inner loss compute: ')
                debug_memory()
            diffopt.step(loss)
            if debug: 
                print(f' {i}th inner loss backwards: ')
                debug_memory()
            
    return f_base_lm, weights, sum(inner_losses)/len(inner_losses)

def compute_lora_outer_loss(base_model, weight_model, batch, inner_lr, outer_grad = True, inner_lora = False, debug = False):
    #adaptation
    
    adapted_base_model, weights, inner_loss = adapt_base_model(base_model, weight_model, batch, inner_lr, inner_lora = inner_lora, outer_grad = outer_grad, debug = debug)
    if debug: 
        print(' Post inner loop: ')
        debug_memory()
        
    set_lora_state(adapted_base_model, True) #turn on lora for the outer loss
    
    with torch.set_grad_enabled(outer_grad):
        #outer_loss = lora_model(base_parameters = adapted_base_model.parameters(), input_ids = batch['qa_toks'], attention_mask = batch['qa_attn_mask'], labels = batch['qa_labels'])['loss']
        
        #original outer loss, this does not lead to memory issues
        #outer_loss = adapted_base_model(input_ids = batch['qa_toks'], attention_mask = batch['qa_attn_mask'], labels = batch['qa_labels'])['loss']
        
        #original outer loss but rewritten as functional form
        outer_loss = adapted_base_model(params = adapted_base_model.parameters(), input_ids = batch['qa_toks'], attention_mask = batch['qa_attn_mask'], labels = batch['qa_labels'])['loss']
    if debug: 
        print(' Post Outer Loss: ')
        debug_memory()
    return outer_loss, inner_loss, adapted_base_model, weights

def validate(base_model, weight_model, val_dataloader, inner_lr, base_state_dict, reset_base_frequency, inner_lora = False):
    outer_losses = []
    inner_losses = []
    weight_list = []
    starting_state_dict = base_model.state_dict() #save the starting state dict so we can reset the model
    base_model.load_state_dict(base_state_dict)
    for i, batch in enumerate(val_dataloader):
        batch = {k: v.to(base_model.device) for k, v in batch.items()}
        outer_loss, inner_loss, adapted_base_model, weights = compute_lora_outer_loss(base_model, weight_model, batch, inner_lr, inner_lora=inner_lora, outer_grad = False)
        outer_losses.append(outer_loss.item())
        inner_losses.append(inner_loss)
        weight_list.append((weights[batch['adaptation_attn_mask']!=0]).detach().cpu().numpy())
        if (i+1) % reset_base_frequency == 0:
            base_model.load_state_dict(base_state_dict)
        else:
            base_model.load_state_dict(adapted_base_model.state_dict())
    base_model.load_state_dict(starting_state_dict)
    cat_weights = np.concatenate(weight_list)
    return {'outer_loss': sum(outer_losses)/len(outer_losses),
            'inner_loss': sum(inner_losses)/len(inner_losses),
            'weight_hist': wandb.Histogram(np_histogram=np.histogram(cat_weights, bins = 100))}
#%%
@hydra.main(config_path='configs', config_name='metatrain_lora_config')
def run(args):

    set_seed(args.seed)
    wandb.init(project='un-camels', entity='nathu', job_type='metatrain_lora', config=args, name=args.notes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Loading models...')
    base_model = get_base_model(args.base_model, args.base_model_state_dict).to(device)
    add_lora(base_model, args.r)
    load_lora(base_model, args.lora_model_path)
    base_state_dict = {k:v.detach().clone().cpu() for k, v in base_model.state_dict().items()}
    
    weight_model = CaMeLS(args.weight_model_base, args.weight_model_freeze_base, nl=args.weight_model_nl).to(device)
    
    optimizer = torch.optim.Adam(list(weight_model.parameters()), lr=args.outer_lr)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir = CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.debug: debug_memory()
    
    print('Loading dataset...')
    
    if args.dataset.name == 'archivalqa':
        train_dataset = ArchivalQADataset(to_absolute_path(args.dataset.train_data_path), tokenizer = tokenizer, max_length = args.dataset.max_length)
        val_dataset = ArchivalQADataset(to_absolute_path(args.dataset.val_data_path), tokenizer = tokenizer, max_length = args.dataset.max_length, downsample_to=50 if args.debug else -1)
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

    accumulated_outer_losses = []
    accumulated_inner_losses = []
    if args.debug: debug_memory()
     
    print('Training...')
    while epoch < args.num_epochs:
       
        if step % 100 == 0:
            print(f' --- Epoch {epoch}, Step {step} ---- ')

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
        
        outer_loss, inner_loss, adapted_base_model, weights = compute_lora_outer_loss(base_model, weight_model, batch, args.inner_lr, outer_grad = True, inner_lora = args.inner_lora, debug = args.debug)
        
        accumulated_outer_losses.append(outer_loss.item())
        accumulated_inner_losses.append(inner_loss)
        
        if args.debug: 
            print(' Post outer loss computation: ', step)
            debug_memory() 
            
        outer_loss = outer_loss / args.gradient_accumulation_steps
        outer_loss.backward()
        with torch.no_grad():
            lora_weight_hash = get_lora_parameters(base_model)[0].abs().sum().item()
        wandb.log({'lora_weight_hash': lora_weight_hash}, commit=False)
        
        if args.debug: 
            print(' Post outer loss backwards: ', step)
            debug_memory() 
        #step the optimizer every gradient_accumulation_steps
        if (step + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(weight_model.parameters(), args.gradient_clip_threshold)  
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({'train':{'outer_loss': np.mean(accumulated_outer_losses),
                        'inner_loss': np.mean(accumulated_inner_losses),
                        'grad_norm': grad_norm}})
            accumulated_outer_losses = []
            accumulated_inner_losses = []
            
        if args.debug: 
            print(' Post Optimizer Step: ')
            debug_memory()
        #update base model parameters, either with the result of the inner step or with the original parameters
        if (step+1) % args.reset_base_frequency == 0:
            base_model.load_state_dict(base_state_dict)
        else:
            base_model.load_state_dict(adapted_base_model.state_dict())
            
       
        del adapted_base_model #does this help with memory?
        if args.debug: 
            print(' Post Base Model State Update: ')
            debug_memory()
        #validation
        if (step+1) % (args.validation_frequency*args.gradient_accumulation_steps) == 0:
            weight_model.eval()
            val_stats = validate(base_model, weight_model, val_dataloader, args.inner_lr, base_state_dict, args.reset_base_frequency, inner_lora=args.inner_lora)
            weight_model.train()
            val_loss = val_stats['outer_loss']
            wandb.log({'val': val_stats})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(weight_model.state_dict(), f'./best_model.pt')
                print('Saved model')
            plot_sample_weights(weight_model, sample_weight_batch, tokenizer, log_to_wandb = True)
        step += 1

if __name__ == '__main__':
    run()
    
#%%
# from hydra import compose, initialize
# from omegaconf import OmegaConf

# initialize(config_path='configs')
# args = compose(config_name="metatrain_lora_config")
# print(OmegaConf.to_yaml(args))
# %%
