"""Embed courses from project_subjects.csv using Gemini embeddings and save to disk."""

#%%
import pandas as pd
import numpy as np

courses_df = pd.read_csv(
    'project_subjects.csv',
    sep=';',
    encoding='utf-8-sig',
    na_values=['NULL'],
    engine='python')

print(courses_df.shape) # 869,556 rows, 15 columns
print(courses_df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 869556 entries, 0 to 869555
# Data columns (total 15 columns):
#  #   Column              Non-Null Count   Dtype  
# ---  ------              --------------   -----  
#  0   project_id          869556 non-null  int64  
#  1   project_name        869556 non-null  object 
#  2   pages               869556 non-null  int64  
#  3   bstype              869556 non-null  int64  
#  4   booktype            869556 non-null  int64  
#  5   fcode               610571 non-null  float64
#  6   fname               610571 non-null  object 
#  7   subject_id          869556 non-null  object 
#  8   parent_subject_id   858051 non-null  object 
#  9   subject_name        869556 non-null  object 
#  10  subject_short_name  857946 non-null  object 
#  11  subject_page        869556 non-null  int64  
#  12  l_key               869556 non-null  int64  
#  13  r_key               869556 non-null  int64  
#  14  level               869556 non-null  int64  
# dtypes: float64(1), int64(8), object(6)
# memory usage: 99.5+ MB

courses_df.head()

#%%
ids = courses_df.project_id.unique()
len(ids) # 11505 unique courses

#%%
# example courses

# 94 topics in the history & law course, id 81
courses_df[courses_df.project_id == 81].subject_short_name.values

# 123 topics in the linear algebra course, id 3827
courses_df[courses_df.project_id == 3827].subject_short_name.values

# # 93 topics in the mathematical analysis course, by МФТИ, id 3316
courses_df[courses_df.project_id == 3316].subject_short_name.values

# %%
# Embed courses and save to disk
from google import genai
from google.genai import types
import numpy as np
import os
from tqdm import tqdm

# Try to load env vars from a .env file if python-dotenv is available
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

embeddings = {}
for (i, id) in tqdm(enumerate(courses_df.project_id.unique()), desc="Embedding courses", total=len(courses_df.project_id.unique())):
    subset = courses_df.loc[courses_df.project_id == id]
    first_row = subset.iloc[0]

    name = courses_df.loc[courses_df.project_id == id, 'project_name'].dropna().iloc[0]
    topics = ', '.join(courses_df.loc[courses_df.project_id == id, 'subject_short_name'].dropna().astype(str).tolist())

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=f"{name}, {topics}",
        config=types.EmbedContentConfig(output_dimensionality=768))  # use 768, 1536, or 3072
    embedding = np.array(response.embeddings[0].values, dtype=float)
    embedding = embedding / np.linalg.norm(embedding)
    embeddings[id] = embedding

    if i > 0 and i % 200 == 0:
        print(f"Processed {i} courses, saving intermediate results...")
        out_dir = "course_embeddings"
        os.makedirs(out_dir, exist_ok=True)
        ids_list = list(embeddings.keys())
        ids_arr = np.array(ids_list, dtype=int)
        vecs = np.vstack([embeddings[i] for i in ids_list])
        np.savez_compressed(os.path.join(out_dir, "course_embeddings.npz"),
                            ids=ids_arr, embeddings=vecs)

out_dir = "course_embeddings"
os.makedirs(out_dir, exist_ok=True)

ids_list = list(embeddings.keys())
ids_arr = np.array(ids_list, dtype=int)
vecs = np.vstack([embeddings[i] for i in ids_list])

# compressed numpy archive with ids and embedding matrix
np.savez_compressed(os.path.join(out_dir, "course_embeddings.npz"),
                    ids=ids_arr, embeddings=vecs)

#%%
# load back
npz_path = os.path.join("course_embeddings", "course_embeddings.npz")
with np.load(npz_path) as data:
    loaded_ids = data["ids"].astype(int)
    vecs = data["embeddings"].astype(float)

print("loaded:", loaded_ids.shape, "ids,", vecs.shape, "embeddings")

# build dict for easy lookup
embeddings_by_id = {int(i): vec for i, vec in zip(loaded_ids, vecs)}

# normalized matrix for fast cosine queries
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
norms[norms == 0] = 1.0
vecs_norm = vecs / norms

def top_k_similar(query_id, k=5):
    if query_id not in embeddings_by_id:
        raise KeyError(f"{query_id} not found in embeddings")
    q_idx = int(np.where(loaded_ids == query_id)[0][0])
    qv = vecs_norm[q_idx]
    sims = vecs_norm @ qv
    # exclude itself
    sims[q_idx] = -np.inf
    top_idx = np.argpartition(-sims, range(k))[:k]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]
    return [(int(loaded_ids[i]), float(sims[i])) for i in top_sorted]

# example: show top 5 similar to the first id in the file
example_id = int(loaded_ids[0])
print("example id:", example_id)
for cid, score in top_k_similar(example_id, k=5):
    print(cid, score)