import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from util import CACHE_DIR

if __name__ == '__main__':
    df = pd.read_csv('/iris/u/nathu/un_camels/data/nyt/meta_train_split.csv')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', cache_folder = CACHE_DIR)
    threshold_frac = 0.9
    save_name = 'meta_train_split_with_matches.csv'
    # Set the batch size
    batch_size = 48

    # Prepare the list to store embeddings
    embeddings = []

    # Iterate through batches of sentences
    for i in tqdm(range(0, len(df['merged_text']), batch_size)):
        
        # Get the batch of sentences
        batch = df['merged_text'][i:i+batch_size]
        
        # Encode the batch of sentences and append the result to embeddings list
        embeddings.append(model.encode(list(batch)))

    # Concatenate the list of embeddings to get a single numpy array

    embeddings = np.vstack(embeddings)
    df['embeddings'] = embeddings.tolist()
    print('Done Computing Embeddings. Pairing Articles...')

    def find_matches(i, embeddings, threshold_frac):
        if i >= len(df)-1:
            return None
        scores = np.einsum('i,ji->j', embeddings[i], embeddings)
        scores[:i+1] = -np.inf
        max_score = np.max(scores)
        matches = np.where(scores > max_score*threshold_frac)[0]
        first_match = matches[0]
        return {
            'match_id': df['doc_id'][first_match],
            'match_score': scores[first_match],
            'max_score': max_score,
            'match_date': df['date'][first_match],
            'match_title': df['title'][first_match],
            'match_text': df['merged_text'][first_match]
        }

    with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust 'max_workers' as per your machine's capacity
        results = list(tqdm(executor.map(find_matches, range(len(df)), [embeddings] * (len(df)), [threshold_frac] * (len(df))), total=len(df)))

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
        
    df.to_csv(f'/iris/u/nathu/un_camels/data/nyt/{save_name}', index = False)