"""Pipeline to extract disciplines and their topics from study plans and work programs."""
#%%
from tqdm import tqdm
import pandas as pd
import src.pipeline_utils as pipeline_utils
import os

def save_rows_to_csv(rows, filename="data/generated/disciplines.csv"):
    if not rows:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    df = pd.DataFrame(rows, columns=[
        "speciality_code","speciality_name","study_plan_url",
        "discipline_name","work_program_url","topics"
    ])
    # append; write header only once
    df.to_csv(filename, index=False, sep=';', mode='a', header=header)

num_study_plans = 1
num_work_programs = 1

#%%
# extract bachelors disciplines: code xx.03.yy
speciality_df = pd.read_csv("data/download/specialities.csv", sep=';')
speciality_df = speciality_df[speciality_df['speciality_code'].str.strip().str.match(r'^\d{2}\.03\.\d{2}$', na=False)]
speciality_df # 187 rows

#%%
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ------------ Config ------------
FLUSH_EVERY_SPECIALITIES = 1
FLUSH_MIN_ROWS = 100
NUM_WORKERS = 4
LOG_FILE = "pipeline.log"
LOG_LEVEL = logging.INFO
# --------------------------------

# Console logging that plays nicely with tqdm
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

# Set up logging: file + console (via tqdm)
logger = logging.getLogger("pipeline")
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

def log(msg, level="info", speciality_code=None, speciality_name=None):
    prefix = f"[{speciality_code} {speciality_name}] " if speciality_code and speciality_name else ""
    if level == "error":
        logger.error(prefix + msg)
    elif level == "warning":
        logger.warning(prefix + msg)
    else:
        logger.info(prefix + msg)

def process_speciality(speciality_code, speciality_name, num_study_plans, num_work_programs):
    log("START", speciality_code=speciality_code, speciality_name=speciality_name)
    local_rows = []

    # 1) Study plans
    try:
        study_plan_urls = pipeline_utils.get_study_plan_urls(speciality_code, speciality_name)
    except Exception as e:
        log(f"FAIL get_study_plan_urls: {e}", level="error",
            speciality_code=speciality_code, speciality_name=speciality_name)
        return local_rows

    used_study_plans = 0
    for study_plan_url in study_plan_urls:
        if used_study_plans >= num_study_plans:
            break

        plan_yielded = False
        try:
            discipline_names = pipeline_utils.extract_discipline_names(study_plan_url, speciality_name)
        except Exception as e:
            log(f"FAIL extract_discipline_names url={study_plan_url}: {e}", level="error",
                speciality_code=speciality_code, speciality_name=speciality_name)
            continue
        if not discipline_names or discipline_names == ['None']:
            log(f"No relevant disciplines in study_plan url={study_plan_url}",
                speciality_code=speciality_code, speciality_name=speciality_name)
            continue

        # 2) For each discipline → work programs (retry-until-success)
        for discipline_name in discipline_names:
            used_work_programs = 0
            try:
                work_program_urls = pipeline_utils.get_work_program_urls(
                    discipline_name, speciality_code, speciality_name
                )
            except Exception as e:
                log(f"FAIL get_work_program_urls discipline='{discipline_name}': {e}", level="error",
                    speciality_code=speciality_code, speciality_name=speciality_name)
                work_program_urls = []

            for work_program_url in work_program_urls:
                if used_work_programs >= num_work_programs:
                    break
                try:
                    topics = pipeline_utils.extract_topics(work_program_url, discipline_name)
                except Exception as e:
                    log(f"FAIL extract_topics url={work_program_url} discipline='{discipline_name}': {e}", level="error",
                        speciality_code=speciality_code, speciality_name=speciality_name)
                    continue
                if not topics or topics == ['None']:
                    log(f"No topics url={work_program_url} discipline='{discipline_name}'",
                        speciality_code=speciality_code, speciality_name=speciality_name)
                    continue

                local_rows.append({
                    "speciality_code": speciality_code,
                    "speciality_name": speciality_name,
                    "study_plan_url": study_plan_url,
                    "discipline_name": discipline_name,
                    "work_program_url": work_program_url,
                    "topics": "; ".join(topics),
                })
                used_work_programs += 1
                plan_yielded = True  # this study plan succeeded at least once

        if plan_yielded:
            used_study_plans += 1

    log(f"DONE → {len(local_rows)} rows", speciality_code=speciality_code, speciality_name=speciality_name)
    return local_rows

def run_pipeline(speciality_df, num_study_plans, num_work_programs, save_rows_to_csv):
    rows_buf, completed, total_written = [], 0, 0

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
                ex.submit(process_speciality, r['speciality_code'], r['speciality_name'],
                          num_study_plans, num_work_programs): (r['speciality_code'], r['speciality_name'])
                for _, r in speciality_df.iterrows()
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Specialities"):
                scode, sname = futures[fut]
                try:
                    batch = fut.result()
                except Exception as e:
                    log(f"FAIL speciality task: {e}", level="error", speciality_code=scode, speciality_name=sname)
                    batch = []
                if batch:
                    rows_buf.extend(batch)

                completed += 1
                if completed % FLUSH_EVERY_SPECIALITIES == 0 or len(rows_buf) >= FLUSH_MIN_ROWS:
                    flush_rows()
    finally:
        # flush even on crash/KeyboardInterrupt
        flush_rows()
        logger.info(f"[DONE] All specialities processed. Total rows written: {total_written}")

run_pipeline(speciality_df, num_study_plans, num_work_programs, save_rows_to_csv)

# %%
