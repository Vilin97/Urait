"""Create a CSV file with unique courses and their topics from the project_subjects.csv file."""
#%%
import pandas as pd
import numpy as np
import json

courses_df = pd.read_csv(
    'data/download/project_subjects.csv',
    sep=';',
    encoding='utf-8-sig',
    na_values=['NULL'],
    engine='python')

agg_dict = {col: 'first' for col in courses_df.columns if col not in ['project_id', 'subject_short_name']}
agg_dict['subject_short_name'] = lambda x: ', '.join(x.dropna().astype(str).unique())
courses_df = courses_df.groupby('project_id', as_index=False).agg(agg_dict)
courses_df = courses_df.rename(columns={'subject_short_name': 'topics', 'fname': 'university'})
courses_df = courses_df.drop(columns=['subject_name', 'subject_id', 'parent_subject_id', 'fcode', 'l_key', 'r_key', 'level', 'subject_page']).reset_index(drop=True)
courses_df

#%%
# add embeddings column
with np.load("course_embeddings/course_embeddings.npz") as data:
    loaded_ids = data["ids"].astype(int)
    embeddings = data["embeddings"].astype(float)
embeddings = embeddings.astype(np.float32)

courses_df['embedding'] = [json.dumps(vec.tolist()) for vec in embeddings]
courses_df.to_csv('data/generated/courses.csv', index=False)

# %%
