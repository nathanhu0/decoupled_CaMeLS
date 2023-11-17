#%%
import hydra
import wandb
from tqdm import tqdm
from util import CACHE_DIR, get_base_model, weighted_lm_loss, set_seed
from transformers import AutoTokenizer, Adafactor
from doc_datasets import NytDataset, ArchivalQADataset
import torch
from hydra.utils import to_absolute_path
import numpy as np
from lora import add_lora, load_lora, get_non_lora_parameters, set_lora_state
from baselines import SalientSpanWeighting, UniformWeighting
from camels import CaMeLS
import csv
import re
import string
from collections import Counter
import json

def adapt_base_model(base_model, dataloader, loss_weighting, lr, optimizer, grad_clip=10):
    if optimizer == 'adafactor':
        optimizer = Adafactor(get_non_lora_parameters(base_model), lr=lr)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(get_non_lora_parameters(base_model), lr=lr)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')
    
    for batch in tqdm(dataloader):
        batch = {k: v.to(base_model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            weights = loss_weighting(input_ids = batch['adaptation_toks'], attention_mask = batch['adaptation_attn_mask'])
        loss = weighted_lm_loss(base_model, batch['adaptation_toks'], batch['adaptation_labels'], batch['adaptation_attn_mask'], weights)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)
        optimizer.step()
        wandb.log({'adaptation_loss': loss.item(), 'grad_norm': grad})
        optimizer.zero_grad()

def evaluate_qa_nll(base_model, dataloader):
    
    losses = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(base_model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = base_model(input_ids = batch['qa_toks'], attention_mask = batch['qa_attn_mask'], labels = batch['qa_labels'])
        loss = outputs.loss
        losses.append(loss.item())
    return np.mean(losses)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

#taken from squad codebase
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def decode_to_clean_text(tokenizer, ids):
    gen_text = tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return list(map(str.strip, gen_text))

def evaluate_qa_f1(base_model, tokenizer, dataloader, output_path, k_generations=1, max_answer_len=24, **generation_kwargs):
    #number of generations is the number of times we generate for each question
    avg_f1s = []
    max_f1s = []
    with open(output_path, 'w', newline='') as writefile:  
        writer = csv.writer(writefile)
        writer.writerow(['question', 'answer', 'generation_num', 'predicted_answer', 'f1_score']) #fix to match the format of the dataset
        for batch in tqdm(dataloader):
            batch = {k: v.to(base_model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if batch['qa_toks'].shape[0] > 1:
                raise ValueError('Batch size must be 1')
            outs = base_model.generate(input_ids = batch['qagen_toks'], 
                                       attention_mask = batch['qagen_attn_mask'], num_return_sequences=k_generations, 
                                       max_length = batch['qagen_toks'].shape[1] + max_answer_len,
                                       pad_token_id = tokenizer.eos_token_id,
                                       **generation_kwargs)
            dec = decode_to_clean_text(tokenizer, outs)
            question = decode_to_clean_text(tokenizer, batch['qagen_toks'])[0]
            answer = batch['qa_answer_text'][0]
            predicted_answers = [s[len(question):] for s in dec]
            f1s = [f1_score(predicted_answer, answer) for predicted_answer in predicted_answers]
            for generation_num, (predicted_answer, f1) in enumerate(zip(predicted_answers, f1s)):
                writer.writerow([question, answer, generation_num, predicted_answer, f1])
            avg_f1s.append(np.mean(f1s))
            max_f1s.append(np.max(f1s))
    return np.mean(avg_f1s), np.mean(max_f1s)
                
            
generation_defaults = {'diversity_penalty': 10.,
                     'num_beam_groups': 4, 
                     'num_beams': 12,
                     'early_stopping': True,
                    }

#%%
@hydra.main(config_path='configs', config_name='eval_config')
def run(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('loading model...')
    base_model = get_base_model(args.base_model, to_absolute_path(args.base_model_state_dict) if args.base_model_state_dict else None)
    add_lora(base_model, args.r)
    load_lora(base_model, to_absolute_path(args.lora_model_path))
    base_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir = CACHE_DIR)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.loss_weighting == 'salient_spans':
        loss_weighting = SalientSpanWeighting(tokenizer=tokenizer)
    elif args.loss_weighting == 'uniform':
        loss_weighting = UniformWeighting()
    # elif args.loss_weighting == 'CaMeLS': #TODO integrate this smoothly with configs
    #     loss_weighting = CaMeLS(args.base_model)
    elif args.loss_weighting == 'init':
        loss_weighting = None
    else:
        raise ValueError(f'Invalid adaptation weighting {args.loss_weighting}')
    
    print('loading dataset...')
    assert args.data_split in ['val', 'test']
    
    if args.dataset.name == 'archivalqa':
        if args.data_split == 'test':
            evaluation_dataset = ArchivalQADataset(to_absolute_path(args.dataset.test_data_path), tokenizer = tokenizer, max_length = args.dataset.max_length, downsample_to=args.downsample_to)
            #same dataset but remove duplicate articles
            adaptation_dataset = evaluation_dataset.get_deduplicated_dataset()
        elif args.data_split == 'val':
            evaluation_dataset = ArchivalQADataset(to_absolute_path(args.dataset.val_data_path), tokenizer = tokenizer, max_length = args.dataset.max_length, downsample_to=args.downsample_to)
            adaptation_dataset = evaluation_dataset.get_deduplicated_dataset()
    else:
        raise NotImplementedError
    
    adaptation_dataloader = torch.utils.data.DataLoader(adaptation_dataset, batch_size=1, shuffle=False, collate_fn=adaptation_dataset.collate_fn)
    evaluation_dataloader = torch.utils.data.DataLoader(evaluation_dataset, batch_size=1, shuffle=False, collate_fn=evaluation_dataset.collate_fn)
    
    wandb.init(project='un-camels', entity='nathu', job_type='evaluation', config=args)
    if args.loss_weighting != 'init':
        print('adapting base model...')
        base_model.train()
        set_lora_state(base_model, False)
        adapt_base_model(base_model, adaptation_dataloader, loss_weighting, args.lr, args.optimizer)
        
    print('evaluating final model...')
    base_model.eval()
    set_lora_state(base_model, True)
    qa_nll = evaluate_qa_nll(base_model, evaluation_dataset)
    max_f1, avg_f1 = evaluate_qa_f1(base_model, tokenizer, evaluation_dataset, './final_generation_outputs.csv', k_generations=args.k_generations, max_answer_len=args.max_answer_len, **generation_defaults)
    wandb.log({'qa_nll': qa_nll, 'max_f1': max_f1, 'avg_f1': avg_f1})
    print(f'QA NLL: {qa_nll}, Max F1: {max_f1}, Avg F1: {avg_f1}')
    with open('./final_qa_metrics.json', 'w') as f:
        json.dump({'qa_nll': qa_nll, 'max_f1': max_f1, 'avg_f1': avg_f1}, f)
    
if __name__ == '__main__':
    run()
    
#%% a useful snippet for debugging
# from hydra import compose, initialize
# from omegaconf import OmegaConf

# initialize(config_path='configs')
# args = compose(config_name="eval_config", overrides=['base_model=distilgpt2', 'lora_model_path=pretrained_models/archivalqa/distilgpt2_lora_r4.pt', 'loss_weighting=uniform'])
# print(OmegaConf.to_yaml(args))
# %%
