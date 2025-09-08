#%%
import pandas as pd
from tqdm import tqdm
import src.url_utils as url_utils
#%%
study_plan_df = pd.read_csv("data/generated/specialities_with_study_plans.csv", delimiter=";")
university_df = pd.read_csv("data/generated/universities_cleaned.csv")

# %%
import numpy as np
import pandas as pd

# --- 1) Prep universities (one row per url_root) ---
u = university_df.copy()
u['uni_name'] = np.where(u['abbreviation'].notna() & (u['abbreviation'] != ''),
                         u['abbreviation'], u['name'])
u = u[['url_root', 'uni_name', 'url']].dropna(subset=['url_root']).rename(columns={'url': 'uni_url'})

# mark ambiguous roots (multiple universities share same root)
u_counts = u.groupby('url_root').size().rename('match_count')
u_first  = (u.sort_values(['url_root', 'uni_name'])
              .drop_duplicates('url_root', keep='first')
              .merge(u_counts, on='url_root', how='left')
              .rename(columns={'url_root': 'doc_root'}))

# --- 2) Prep study plans (compute roots once, then map) ---
sp = study_plan_df.copy()

# cache extract_root over unique URLs (faster if extract_root is non-trivial)
uniq_urls = sp['study_plan_url'].dropna().unique()
roots_map = {url: url_utils.extract_root(url) for url in uniq_urls}
sp['doc_root'] = sp['study_plan_url'].map(roots_map)

# --- 3) Join & outputs ---
matched = sp.merge(u_first, on='doc_root', how='left', indicator=True)

unmatched = matched.loc[(matched['_merge'] == 'left_only') & matched['study_plan_url'].notna(),
                        'study_plan_url'].tolist()

# rows whose root maps to multiple universities (optional: inspect these)
ambiguous = (matched.loc[matched['match_count'].fillna(0) > 1, ['study_plan_url','doc_root']]
                   .drop_duplicates()
                   .merge(u.rename(columns={'url_root':'doc_root'}), on='doc_root', how='left')
                   .sort_values(['study_plan_url','uni_name']))

print(f"Total unmatched study_plan URLs: {len(unmatched)}")

#%%
matched