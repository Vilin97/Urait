"""Make embeddings for disciplines."""
#%%
import pandas as pd
import json
from tqdm import tqdm
import src.utils as utils

#%%
# make disciplines embeddings
disciplines_df = pd.read_csv('data/generated/disciplines.csv', sep=';')
disciplines_df['topics'] = disciplines_df['topics'].apply(lambda s: s.replace('\n', '; ').replace('*  ', '; '))

client = utils.get_gemini_client()
embeddings = []
for (i, row) in tqdm(disciplines_df.iterrows(), desc="Embedding disciplines", total=len(disciplines_df)):
    speciality = row['speciality_name']
    name = row['discipline_name']
    topics = row['topics']
    text = f"{speciality}, {name}, {topics}"
    embedding = utils.embed_text(text, client)
    embeddings.append(embedding)
    
disciplines_df['embedding'] = [json.dumps(vec.tolist()) for vec in embeddings]
disciplines_df.to_csv('data/generated/disciplines_with_embeddings.csv', index=False, sep=';')
