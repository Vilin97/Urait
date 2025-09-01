"""Make embeddings for disciplines."""
#%%
import pandas as pd
import json
from tqdm import tqdm
import src.utils as utils
import matplotlib.pyplot as plt
import numpy as np

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

#%%
# load disciplines with embeddings
disciplines_df = pd.read_csv('data/generated/disciplines_with_embeddings.csv', sep=';')
disciplines_df['embedding'] = disciplines_df['embedding'].apply(lambda s: np.array(json.loads(s), dtype=np.float32))

# count topics (ignore empty items after splitting)
topic_counts = disciplines_df['topics'].apply(lambda s: len([t for t in s.split(';') if t.strip() != ""]))

# quick stats (optional)
print("Topics per discipline — min:", topic_counts.min(), "max:", topic_counts.max(), "median:", topic_counts.median())

top_idx = topic_counts.nlargest(20).index
disciplines_df = disciplines_df.drop(index=top_idx).reset_index(drop=True)

# recompute topic_counts after dropping
topic_counts = disciplines_df['topics'].apply(lambda s: len([t for t in s.split(';') if t.strip() != ""]))

# histogram
plt.figure(figsize=(8, 4))
max_tc = int(topic_counts.max())
bins_total = 40

bins = np.linspace(0, max_tc + 1, min(max_tc + 2, bins_total + 1))  # ensure at least 1 count per bin
n, bins_edges, patches = plt.hist(topic_counts, bins=bins, align='left', edgecolor='black')
plt.xlabel(f'Number of topics, bin width: {(bins[1]-bins[0]):.1f}')
plt.ylabel('Number of disciplines')
plt.title('Distribution of topics per discipline')
plt.grid(axis='y', alpha=0.6)

# annotate counts above bars in small red numbers
for count, patch in zip(n, patches):
    if count <= 0:
        continue
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_height()
    offset = max(n) * 0.01 if max(n) > 0 else 0.1
    plt.text(x, y + offset, str(int(count)), ha='center', va='bottom', color='red', fontsize=8)

plt.tight_layout()
plt.savefig('data/generated/topics_histogram.png', dpi=150)
plt.show()


# zoomed histogram [0,20] with 1 bin per integer
plt.figure(figsize=(8, 4))
min_val, max_val = 0, 20
bins = np.arange(min_val, max_val + 2)  # edges 0..11 to get bins for 0..20
filtered = topic_counts[(topic_counts >= min_val) & (topic_counts <= max_val)]

n, bins_edges, patches = plt.hist(filtered, bins=bins, align='left', edgecolor='black')
plt.xlabel('Number of topics')
plt.ylabel('Number of disciplines')
plt.title(f'Distribution of topics per discipline (0–{max_val})')
plt.xticks(np.arange(min_val, max_val + 1))
plt.grid(axis='y', alpha=0.6)

# annotate counts above bars
for count, patch in zip(n, patches):
    if count <= 0:
        continue
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_height()
    offset = max(n) * 0.02 if max(n) > 0 else 0.1
    plt.text(x, y + offset, str(int(count)), ha='center', va='bottom', color='red', fontsize=8)

plt.tight_layout()
plt.savefig('data/generated/topics_histogram_0_10.png', dpi=150)
plt.show()