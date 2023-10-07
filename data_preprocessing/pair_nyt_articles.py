import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('../')
sys.path.append('./')
from util import CACHE_DIR
import argparse
from concurrent.futures import ThreadPoolExecutor
import os


def main(args):
    df = pd.read_csv(args.data_path).reset_index(drop=True)
    model = SentenceTransformer(args.model, cache_folder=CACHE_DIR)
    threshold_frac = args.threshold_frac
    save_path = args.save_path
    batch_size = 48

    embeddings = []
    for i in tqdm(range(0, len(df['text']), batch_size)):
        batch = df['text'][i:i+batch_size]
        embeddings.append(model.encode(list(batch)))

    embeddings = np.vstack(embeddings)
    df['embeddings'] = embeddings.tolist()
    print('Done Computing Embeddings. Pairing Articles...')

    def find_matches(i, embeddings, threshold_frac):
        if i >= len(df):
            return None
        scores = np.einsum('i,ji->j', embeddings[i], embeddings)
        
        date_mask = df['date'] > df['date'][i]
        scores[~date_mask] = -np.inf
        max_score = np.max(scores)
        matches = np.where(scores > max_score*threshold_frac)[0]
        if len(matches) == 0:
            return None
        first_match = matches[0]
        return {
            'match_id': df['doc_id'][first_match],
            'match_score': scores[first_match],
            'max_score': max_score,
            'match_date': df['date'][first_match],
            'match_title': df['title'][first_match],
            'match_text': df['text'][first_match]
        }

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(find_matches, range(len(df)), [embeddings] * len(df), [threshold_frac] * len(df)), total=len(df)))

    df['match_id'] = np.nan
    df['match_score'] = np.nan
    df['max_score'] = np.nan
    df['match_date'] = np.nan
    df['match_title'] = np.nan
    df['match_text'] = np.nan

    for i, result in enumerate(results):
        if result is not None:
            df.loc[i, 'match_id'] = result['match_id']
            df.loc[i, 'match_score'] = result['match_score']
            df.loc[i, 'max_score'] = result['max_score']
            df.loc[i, 'match_date'] = result['match_date']
            df.loc[i, 'match_title'] = result['match_title']
            df.loc[i, 'match_text'] = result['match_text']
        
    df.to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L12-v2', help='Path to the model')
    parser.add_argument('--save_path', type=str, default='./meta_train_split_with_matches.csv', help='Name of the save file')
    parser.add_argument('--threshold_frac', type=float, default=0.9, help='Threshold fraction')
    parser.add_argument('--data_path', type=str, default='/iris/u/nathu/un_camels/data/nyt/meta_train_split.csv', help='Path to data file')

    args = parser.parse_args()
    main(args)
