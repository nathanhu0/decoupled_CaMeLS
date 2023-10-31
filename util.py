#%%
import os
import getpass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List
from accelerate import load_checkpoint_and_dispatch, init_empty_weights


if os.path.exists('/scr/scr-with-most-space'):
    CACHE_DIR = '/scr/scr-with-most-space/' + getpass.getuser()
elif os.path.exists('/scr-ssd'):
    CACHE_DIR = '/scr-ssd/' + getpass.getuser()
else:
    CACHE_DIR = f'/scr/{getpass.getuser()}/cache'
    
def get_base_model(base_model, base_model_state_dict = None):   
    model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir = CACHE_DIR, device_map = 'auto')
    if base_model_state_dict is not None:
        print('Loading base model state dict...')
        model.load_state_dict(torch.load(base_model_state_dict, map_location=model.device))
    return model

# def get_base_model(base_model, base_model_state_dict = None):
#     if base_model_state_dict is not None:
#         with init_empty_weights():
#             model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir = CACHE_DIR)
#         model = load_checkpoint_and_dispatch(model, base_model_state_dict, device_map='auto')
#     else:
#         model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir = CACHE_DIR, device_map = 'auto')
#     return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def weighted_lm_loss(model, input_ids, target_ids, attention_mask, weights):
    outputs = model(input_ids=input_ids,
                attention_mask=attention_mask,
                labels = target_ids
            )
    batch_size = len(input_ids)
    #reshape logits and labels to be (batch_size*(seq_len-1), vocab_size)
    #we drop the last logit as there are no labels, and the first label as there are no logits
    reshaped_logits = outputs.logits[:, :-1, :].reshape(-1, outputs.logits.shape[-1])
    reshaped_labels = target_ids[:, 1:].reshape(-1)
    l = F.cross_entropy(reshaped_logits, reshaped_labels, ignore_index = -100, reduction = 'none')
    return (l.reshape(batch_size, -1)*weights[:, 1:]*attention_mask[:, 1:]).mean()


def create_colored_text(words: List[str], data: List[float], font_path='DejaVuSansMono.ttf', pos_cmap=None, neg_cmap=None, max_intensity=.8) -> Image:
    # Create a colormap that maps your data range to colors
    
    if pos_cmap is None:
        pos_cmap = plt.cm.get_cmap('Reds')
    if neg_cmap is None:
        neg_cmap = plt.cm.get_cmap('Blues')
    max_mag = max(abs(np.min(data)), np.max(data))
    cmap = lambda x: pos_cmap(x*max_intensity/max_mag) if x > 0 else neg_cmap(-x*max_intensity/max_mag)
    max_width = 800
    line_height = 25
    
    # Set the font
    font = ImageFont.truetype(font_path, 16)
    # Find the maximum font size of all the words
    max_font_size = max([font.getbbox(word)[3] for word in words])
    # Initialize the x- and y-coordinates for drawing the words
    x = 0
    y = 0
    
    for word in words:
        word_width = font.getlength(word)
        if x + word_width > max_width:
            # Move to the next line
            x = 0
            y += line_height
        x += word_width
    
    final_height = y + line_height
    # Create a new image with a white background
    image = Image.new('RGB', (max_width, final_height), (255, 255, 255))
    # Get a drawing context
    draw = ImageDraw.Draw(image)
    x = 0
    y = 0
    # Iterate over the words in the text passage
    for i, word in enumerate(words):
        # Get the numeric value for the current word
        value = data[i]
        # Map the numeric value to a color from the colormap
        color = cmap(value)
        # Get the color in 8-bit RGB format
        rgb_color = tuple(int(c * 255) for c in color[:3])
        word_width = font.getlength(word)
        # Check if the word fits on the current line
        if x + word_width > max_width:
            # Move to the next line
            x = 0
            y += line_height
        # Draw the word with the mapped color and black foreground color
        draw.rectangle([(x, y), (x + word_width, y + max_font_size)], fill=rgb_color)
        draw.text((x, y), word, font=font, fill=(0, 0, 0))
        # Increment the x-coordinate for drawing the next word
        x += word_width
    image = image.crop((0, 0, max_width, y + line_height)).resize((max_width, y + line_height))
    return image


# %%

def shuffle_groups(df, group_col):
    """
    Shuffles the order of groups in a Pandas DataFrame without shuffling the order of items within each group.

    Parameters:
    - df: the input DataFrame
    - group_col: the name of the column containing the groups to be shuffled

    Returns:
    - a shuffled copy of the input DataFrame
    """
    # Get a list of unique groups
    groups = df[group_col].unique()

    # Shuffle the list of groups
    np.random.shuffle(groups)

    # Define a sorting key that sorts by the shuffled order of groups
    def sort_key(row):
        return np.argwhere(groups == row[group_col])[0][0]

    df['temp'] = df.apply(sort_key, axis=1)
    shuffled_df = df.sort_values('temp', kind='stable').drop('temp', axis=1).reset_index(drop=True)
    return shuffled_df

#given a pd dataframe, return a head of the dataframe such that column column has k unique values
def return_k_unique(df, k, column): 
    if k >= len(df[column].unique()):
        return df
    else:
        values_to_keep = df[column].unique()[:k]
        return df[df.apply(lambda x: x[column] in values_to_keep, axis=1)]
# %%
