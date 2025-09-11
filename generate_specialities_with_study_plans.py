#%% Imports
import os
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re

import src.pipeline_utils as pipeline_utils
import src.url_utils as url_utils

#%% Config
OUTPUT_CSV = "data/generated/study_plans_2.csv"
NUM_STUDY_PLANS = 50
NUM_WORKERS = 4
FLUSH_MIN_ROWS = 5
LOG_FILE = "pipeline_study_plans.log"
LOG_LEVEL = logging.INFO

#%% Logging (console-friendly with tqdm)
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

logger = logging.getLogger("pipeline_study_plans")
logger.setLevel(LOG_LEVEL)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(fmt)
fh.setLevel(LOG_LEVEL)

ch = TqdmLoggingHandler()
ch.setFormatter(fmt)
ch.setLevel(LOG_LEVEL)

logger.addHandler(fh)
logger.addHandler(ch)
logger.propagate = False  # avoid double-printing via root

def log(msg, level="info", speciality_name=""):
    to_log = f"{msg} | {speciality_name}"
    (logger.error if level == "error" else logger.warning if level == "warning" else logger.info)(to_log)

#%% IO helpers
def save_rows_to_csv(rows, filename=OUTPUT_CSV):
    if not rows:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    df = pd.DataFrame(rows, columns=[
        "speciality_code", "speciality_name", "university",
        "study_plan_url", "disciplines"
    ])
    df.to_csv(filename, index=False, sep=';', mode='a', header=header)

#%% Load and prepare speciality + university pairs
speciality_df = pd.read_csv("data/download/specialities.csv", sep=';')
speciality_df = speciality_df[
    speciality_df['speciality_code'].astype(str).str.strip().str.match(r'^\d{2}\.03\.\d{2}$', na=False)
]

university_df = pd.read_csv("data/generated/universities_cleaned.csv")

#%% Per-row processing
def process_speciality_row(row):
    scode = row['speciality_code']
    sname = row['speciality_name']
    local_rows = []

    idx = row.name
    log(f"[START] [{idx}]", speciality_name=sname)
    try:
        study_plan_urls = pipeline_utils.get_study_plan_urls(scode, sname)
    except Exception as e:
        log(f"FAIL get_study_plan_urls: {e}", level="error", speciality_name=sname)
        return local_rows

    log(f"        [{idx}] Found {len(study_plan_urls)} study plan URLs", speciality_name=sname)
    used = 0
    for url in study_plan_urls:
        if used >= NUM_STUDY_PLANS:
            break
        try:
            disciplines = pipeline_utils.extract_discipline_names(url, sname) or []
        except Exception as e:
            log(f"            [{idx}] FAIL extract_discipline_names url={url}: {e}", level="error", speciality_name=sname)
            continue

        # normalize and filter
        disciplines = [d.strip() for d in disciplines if d and d.strip().lower() != 'none']
        if not disciplines:
            log(f"            [{idx}] Extracted no disciplines from {url}, skipping", level="warning", speciality_name=sname)
            continue
        elif len(disciplines) <= 1:
            log(f"            [{idx}] Extracted only discipline '{disciplines[0]}' from {url}, skipping", level="warning", speciality_name=sname)
            continue
        elif len(disciplines) <= 5:
            log(f"            [{idx}] Extracted few disciplines {disciplines} from {url}", level="warning", speciality_name=sname)
        else:
            log(f"            [{idx}] Extracted {len(disciplines)} disciplines from {url}", speciality_name=sname)

        # Find matching university
        study_plan_root = url_utils.extract_root(url)
        matched_unis = university_df[university_df['url_root'] == study_plan_root]
        if len(matched_unis) == 1:
            uni = matched_unis.iloc[0]['name']
        elif len(matched_unis) > 1:
            uni = matched_unis.iloc[0]['name']
            log(f"            [{idx}] Ambiguous university match for study_plan url={url}: {matched_unis['name'].tolist()}",
                level="warning", speciality_name=sname)
        else:
            uni = "Unknown"
            log(f"            [{idx}] No matching university found for study_plan url={url}",
                level="warning", speciality_name=sname)

        local_rows.append({
            "speciality_code": scode,
            "speciality_name": sname,
            "university": uni,
            "study_plan_url": url,
            "disciplines": "; ".join(disciplines),
        })
        used += 1

    if used == 0:
        log(f"            [{idx}] No usable study plans found", level="warning", speciality_name=sname)
    elif used < NUM_STUDY_PLANS:
        log(f"            [{idx}] Used {used}/{NUM_STUDY_PLANS} study plans", level="warning", speciality_name=sname)

    log(f"[DONE]  [{idx}] → {len(local_rows)} rows", speciality_name=sname)
    return local_rows

#%% Orchestration
def run_pipeline(df):
    rows_buf, total_written = [], 0

    def flush_rows():
        nonlocal rows_buf, total_written
        if not rows_buf:
            return
        try:
            n = len(rows_buf)
            save_rows_to_csv(rows_buf)
            total_written += n
            logger.info(f"[WRITE] +{n} rows (cumulative={total_written})")
        except Exception as e:
            logger.error(f"[WRITE-FAIL] Could not write {len(rows_buf)} rows: {e}")
        finally:
            rows_buf.clear()

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {
                ex.submit(process_speciality_row, r): (r['speciality_code'], r['speciality_name'])
                for _, r in df.iterrows()
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Specialities"):
                scode, sname = futures[fut]  # fixed unpacking (removed 'uni')
                try:
                    batch = fut.result()
                except Exception as e:
                    log(f"FAIL speciality task: {e}", level="error", speciality_name=sname)
                    batch = []
                if batch:
                    rows_buf.extend(batch)
                if len(rows_buf) >= FLUSH_MIN_ROWS:
                    flush_rows()
    finally:
        flush_rows()
        logger.info("[DONE] specialities_with_study_plans complete")

#%% Run
if __name__ == "__main__":
    # Clear output file on each run to avoid accidental appends across runs.
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    run_pipeline(speciality_df)

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
