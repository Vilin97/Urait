#%%
# Compute statistics
study_plans_df = pd.read_csv(OUTPUT_CSV, sep=';')
print(f"Generated {len(study_plans_df)} study plans for {len(study_plans_df['speciality_name'].unique())} specialities and {len(study_plans_df['university'].unique())} unique universities.")

# Count total and unique disciplines
disc_series = study_plans_df['disciplines'].dropna().astype(str).str.split(r'\s*[;\.]\s*').explode().str.strip()
disc_series = disc_series[disc_series != '']
total_disciplines = len(disc_series)
unique_disciplines = disc_series.str.lower().unique()
num_unique_disciplines = len(unique_disciplines)
print(f"Total discipline occurrences: {total_disciplines}")
print(f"Unique disciplines (case-insensitive): {num_unique_disciplines}")

#%%
# Aggregate by discipline
university_df = pd.read_csv("data/generated/universities_cleaned.csv")
df = pd.read_csv("data/generated/study_plans.csv", sep=';')

# explode disciplines, map students from university_df, group & collect
uamt = pd.to_numeric(university_df['students_amount'], errors='coerce').fillna(0).astype(int)
m = dict(zip(university_df['name'].astype(str).str.strip(), uamt))
m.update(dict(zip(university_df['abbreviation'].dropna().astype(str).str.strip(),
                  uamt[university_df['abbreviation'].notna()])))

w = df.copy()
w['discipline'] = w['disciplines'].fillna('').str.split(r'\s*[;\.]\s*', regex=True)
w = w.explode('discipline').drop_duplicates()
w['discipline'] = w['discipline'].fillna('').str.strip().str.lower()
w = w[w['discipline'].str.len() > 1]
w['students_amount'] = w['university'].astype(str).str.strip().map(m).fillna(0).astype(int)
w.sort_values('students_amount', ascending=False).drop(columns=['disciplines']).to_csv("data/generated/disciplines_all.csv", index=False, sep=';')

g = (w.groupby('discipline', sort=False)
       .agg(total_students_amount=('students_amount','sum'),
            speciality_name=('speciality_name', list),
            speciality_code=('speciality_code', list),
            university=('university', list),
            study_plan_urls=('study_plan_url', list))
       .reset_index())

g['num_speciality_name'] = g['speciality_name'].apply(lambda xs: sum(pd.notna(xs)))
g['num_university'] = g['university'].apply(lambda xs: sum(pd.notna(xs)))
for c in ['speciality_name','speciality_code','university','study_plan_urls']:
    g[c] = g[c].apply(lambda xs: list(dict.fromkeys([v for v in xs if pd.notna(v) and str(v).strip()])))
g['num_distinct_speciality_name'] = g['speciality_name'].apply(len)
g['num_distinct_university'] = g['university'].apply(len)

disciplines_grouped_df = g.sort_values('total_students_amount', ascending=False).reset_index(drop=True)
disciplines_grouped_df

#%%
# Add search counts
query_df = pd.read_csv("data/download/search_queries.csv", usecols=["query","search_count"]).fillna({"query":""})
query_df['query'] = query_df['query'].astype(str).str.strip().str.lower()

disciplines_grouped_df = disciplines_grouped_df.merge(
    query_df, left_on='discipline', right_on='query', how='left'
).drop(columns=['query']).fillna({'search_count': 0}).astype({'search_count': int})

disciplines_grouped_df.sort_values("search_count", ascending=False)
disciplines_grouped_df.to_csv("data/generated/disciplines_by_popularity.csv", index=False, sep=';')
# %%
