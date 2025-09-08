#%%
import pandas as pd
from tqdm import tqdm

#%%
study_plan_df = pd.read_csv("data/generated/specialities_with_study_plans.csv", delimiter=";")
university_df = pd.read_csv("data/generated/universities_cleaned.csv")

#%%
import src.url_utils as url_utils

unmatched = []
for idx, row in tqdm(study_plan_df.iterrows(), total=len(study_plan_df), desc="Processing study plans"):
    # print(f"[{idx}] {row['speciality_name']}: {url_utils.extract_root(row['study_plan_url'])} {row['study_plan_url']}")
    url = row['study_plan_url']

    uni = "Unknown"
    matches = []
    if not url or pd.isna(url):
        print(f"study_plan_url is empty or NaN for idx={idx}")
    else:
        doc_root = url_utils.extract_root(url)
        for _, urow in university_df.iterrows():
            uni_root = urow.get("url_root")
            if doc_root == uni_root:
                uni_name = urow.get("abbreviation") if pd.notna(urow.get("abbreviation")) else urow.get("name")
                matches.append((uni_name, uni_root, urow['url']))

    if len(matches) == 0:
        unmatched.append(url)
        print(f"    No matching university found for study_plan url={url}")
    elif len(matches) > 1:
        print(f"Multiple ({len(matches)}) matching universities for study_plan url={url}:")
        for abbr, uni_root, uurl in matches:
            print(f" - {abbr}, root={uni_root}, url={uurl}")
    else:
        uni, uni_root, uni_url = matches[0]
        # print(f"Matching university: {uni}, url={uni_url}, root={uni_root}")
print(f"Total unmatched study_plan URLs: {len(unmatched)}")
unmatched

# %%
