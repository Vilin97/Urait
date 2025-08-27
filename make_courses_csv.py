#%%

import pandas as pd
import numpy as np

#%%
courses_df = pd.read_csv(
    'project_subjects.csv',
    sep=';',
    encoding='utf-8-sig',
    na_values=['NULL'],
    engine='python')

agg_dict = {col: 'first' for col in courses_df.columns if col not in ['project_id', 'subject_short_name']}
agg_dict['subject_short_name'] = lambda x: ', '.join(x.dropna().astype(str).unique())
courses_df = courses_df.groupby('project_id', as_index=False).agg(agg_dict)
courses_df = courses_df.rename(columns={'subject_short_name': 'topics', 'fname': 'university'})
courses_df = courses_df.drop(columns=['subject_name', 'subject_id', 'parent_subject_id', 'fcode', 'l_key', 'r_key', 'level', 'subject_page']).reset_index(drop=True)

courses_df.to_csv('courses.csv', index=False)
#%%
courses_df['subject_short_name'][0]
