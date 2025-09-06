#%%
import src.google_search as google_search
import pandas as pd
from tqdm import tqdm

#%%
universities_df = pd.read_csv('data/download/partners.csv')

#%%
# find university websites
urls = []
for (i, row) in tqdm(universities_df.iterrows(), total=len(universities_df)):
    university_name = row['name']
    query = f"{university_name}"
    try:
        results = google_search.search(query, rate_limit=0)
        url = results[0]['url']
    except Exception as e:
        print(f"Error for {university_name}: {e}")
        url = ""
    urls.append(url)
    print(row['abbreviation'], url)

universities_df['url'] = urls
universities_df.to_csv('data/generated/universities.csv', index=False)