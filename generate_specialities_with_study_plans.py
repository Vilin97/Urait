#%% Imports
import os
import re
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import src.pipeline_utils as pipeline_utils

#%% Config
OUTPUT_CSV = "data/generated/specialities_with_study_plans.csv"
NUM_STUDY_PLANS = 1
NUM_WORKERS = 4
FLUSH_MIN_ROWS = 20
LOG_FILE = "pipeline_study_plans.log"
LOG_LEVEL = logging.INFO

#%% Logging (console-friendly with tqdm)
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
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

def log(msg, level="info", speciality_name='', university=''):
    university_short = university.split(' ')[0]
    to_log = f"{msg} | {speciality_name} | {university_short}"
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
speciality_df = speciality_df[speciality_df['speciality_code'].astype(str).str.strip().str.match(r'^\d{2}\.03\.\d{2}$', na=False)]

university_df = pd.read_csv("data/download/selected_universities.csv")
university_df.rename(columns={'speciality': 'speciality_name', 'vuz': 'university'}, inplace=True)
university_df = (
    university_df
    .dropna(subset=['speciality_name', 'university'])
    [['speciality_name', 'university']]
    .drop_duplicates()
)

speciality_df = speciality_df.merge(university_df, on='speciality_name', how='inner')
speciality_df = speciality_df[['speciality_code', 'speciality_name', 'university']]
speciality_df = speciality_df
print(speciality_df.head())

#%% Per-row processing
def process_speciality_row(row):
    scode = row['speciality_code']
    sname = row['speciality_name']
    uni = row['university']
    local_rows = []

    idx = row.name
    log(f"[START] [{idx}]", speciality_name=sname, university=uni)
    try:
        study_plan_urls = pipeline_utils.get_study_plan_urls(scode, sname, uni)
    except Exception as e:
        log(f"FAIL get_study_plan_urls: {e}", level="error", speciality_name=sname, university=uni)
        return local_rows

    log(f"        [{idx}] Found {len(study_plan_urls)} study plan URLs", speciality_name=sname, university=uni)
    used = 0
    for url in study_plan_urls:
        if used >= NUM_STUDY_PLANS:
            break
        try:
            disciplines = pipeline_utils.extract_discipline_names(url, sname) or []
            log(f"            [{idx}] Extracted {len(disciplines)} disciplines from {url}", speciality_name=sname, university=uni)
        except Exception as e:
            # log(f"            [{idx}] FAIL extract_discipline_names url={url}: {e}", level="error", speciality_name=sname, university=uni)
            continue

        # normalize and filter
        disciplines = [d.strip() for d in disciplines if d and d.strip().lower() != 'none']
        if not disciplines or len(disciplines) <= 1:
            log(f"            [{idx}] No relevant disciplines in study_plan url={url}", speciality_name=sname, university=uni)
            continue

        local_rows.append({
            "speciality_code": scode,
            "speciality_name": sname,
            "university": uni,
            "study_plan_url": url,
            "disciplines": "; ".join(disciplines),
        })
        used += 1
    if used < NUM_STUDY_PLANS:
        log(f"            [{idx}] Could not find any study plans", speciality_name=sname, university=uni)

    log(f"[DONE]  [{idx}] → {len(local_rows)} rows", speciality_name=sname, university=uni)
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
                ex.submit(process_speciality_row, r): (r['speciality_code'], r['speciality_name'], r['university'])
                for _, r in df.iterrows()
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Specialities"):
                scode, sname, uni = futures[fut]
                try:
                    batch = fut.result()
                except Exception as e:
                    log(f"FAIL speciality task: {e}", level="error", speciality_name=sname, university=uni)
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

# %%
