#%% Imports
import os
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import src.pipeline_utils as pipeline_utils

#%% Config
INPUT_CSV = "data/generated/specialities_with_study_plans.csv"
OUTPUT_CSV = "data/generated/disciplines_with_work_programs.csv"
NUM_WORK_PROGRAMS = 1
NUM_WORKERS = 8
FLUSH_MIN_ROWS = 100
LOG_FILE = "pipeline_work_programs.log"
LOG_LEVEL = logging.INFO

#%% Logging
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

logger = logging.getLogger("pipeline_work_programs")
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

def log(msg, level="info", speciality_code=None, speciality_name=None, university=None, discipline=None):
    parts = []
    if speciality_code and speciality_name:
        parts.append(f"{speciality_code} {speciality_name}")
    if university:
        parts.append(university)
    if discipline:
        parts.append(f"'{discipline}'")
    prefix = f"[{' | '.join(parts)}] " if parts else ""
    (logger.error if level == "error" else logger.warning if level == "warning" else logger.info)(prefix + msg)

#%% IO helpers
def save_rows_to_csv(rows, filename=OUTPUT_CSV):
    if not rows:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    df = pd.DataFrame(rows, columns=[
        "speciality_code", "speciality_name", "university",
        "discipline_name", "work_program_url", "topics"
    ])
    df.to_csv(filename, index=False, sep=';', mode='a', header=header)

#%% Load disciplines from the first CSV and expand
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Required input not found: {INPUT_CSV}")

sp_df = pd.read_csv(INPUT_CSV, sep=';')

# Expand semicolon-separated disciplines into rows
rows = []
for _, r in sp_df.iterrows():
    disc_field = str(r.get('disciplines', '') or '')
    parts = [d.strip() for d in disc_field.split(';') if d and d.strip()]
    parts = [d for d in parts if d.lower() != 'none']
    for d in parts:
        rows.append({
            "speciality_code": r['speciality_code'],
            "speciality_name": r['speciality_name'],
            "university": r['university'],
            "discipline_name": d
        })

disc_df = pd.DataFrame(rows, columns=["speciality_code", "speciality_name", "university", "discipline_name"])
# Remove duplicates to avoid redundant searches
disc_df = disc_df.drop_duplicates().reset_index(drop=True)

#%% Per-discipline processing
def process_discipline(row):
    scode = row['speciality_code']
    sname = row['speciality_name']
    uni = row['university']
    dname = row['discipline_name']
    local_rows = []

    try:
        wp_urls = pipeline_utils.get_work_program_urls(dname, scode, sname, uni) or []
    except Exception as e:
        log(f"FAIL get_work_program_urls: {e}", level="error", speciality_code=scode, speciality_name=sname, university=uni, discipline=dname)
        return local_rows

    used = 0
    seen = set()
    for url in wp_urls:
        if used >= NUM_WORK_PROGRAMS:
            break
        if not url or url in seen:
            continue
        seen.add(url)

        try:
            topics = pipeline_utils.extract_topics(url, dname) or []
        except Exception as e:
            log(f"FAIL extract_topics url={url}: {e}", level="error", speciality_code=scode, speciality_name=sname, university=uni, discipline=dname)
            continue

        topics = [t.strip() for t in topics if t and t.strip().lower() != 'none']
        if not topics:
            log(f"No topics url={url}", speciality_code=scode, speciality_name=sname, university=uni, discipline=dname)
            continue

        local_rows.append({
            "speciality_code": scode,
            "speciality_name": sname,
            "university": uni,
            "discipline_name": dname,
            "work_program_url": url,
            "topics": "; ".join(topics),
        })
        used += 1

    if local_rows:
        log(f"DONE {len(local_rows)}", speciality_code=scode, speciality_name=sname, university=uni, discipline=dname)
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
                ex.submit(process_discipline, r): (r['speciality_code'], r['speciality_name'], r['university'], r['discipline_name'])
                for _, r in df.iterrows()
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Disciplines"):
                scode, sname, uni, dname = futures[fut]
                try:
                    batch = fut.result()
                except Exception as e:
                    log(f"FAIL discipline task: {e}", level="error", speciality_code=scode, speciality_name=sname, university=uni, discipline=dname)
                    batch = []
                if batch:
                    rows_buf.extend(batch)
                if len(rows_buf) >= FLUSH_MIN_ROWS:
                    flush_rows()
    finally:
        flush_rows()
        logger.info("[DONE] disciplines_with_work_programs complete")

#%% Run
if __name__ == "__main__":
    # Clear output file on each run to avoid accidental appends across runs.
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    run_pipeline(disc_df)
