#%%
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
queries = ["Which planet is known as the Red Planet?", "Which planet is the largest in our solar system?"]
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(queries, prompt_name="Retrieval")
document_embeddings = model.encode_document(documents, prompt_name="Retrieval")
print(query_embeddings.shape, document_embeddings.shape)
# (2, 768) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)

# %%
queries = ["Which planet is known as the Red Planet?", "Which planet is the largest in our solar system?"]
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (2, 768) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)

#%%
from sentence_transformers import SentenceTransformer
import pandas as pd
df = pd.read_csv("data/generated/disciplines_all.csv", sep=';')
s1 = 'Бизнес-информатика'
# s2 = 'Математика'
s2 = 'Искусства и гуманитарные науки'
df1 = df[df['speciality_name'] == s1]
df2 = df[df['speciality_name'] == s2]

model = SentenceTransformer("google/embeddinggemma-300m", default_prompt_name='Retrieval')

#%%
import torch
n = 32
print(df1['discipline'].tolist()[:n])
emb1 = model.encode(df1['discipline'].tolist()[:n])
print(df2['discipline'].tolist()[:n])
emb2 = model.encode(df2['discipline'].tolist()[:n])

sim11 = model.similarity(emb1, emb1)
sim22 = model.similarity(emb2, emb2)
sim12 = model.similarity(emb1, emb2)
print(sim11.mean(), sim22.mean(), sim12.mean())
for name, arr in (("sim11", sim11), ("sim22", sim22), ("sim12", sim12)):
    arr = arr[~torch.eye(arr.shape[0], dtype=bool)]
    print(f"{name}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}")