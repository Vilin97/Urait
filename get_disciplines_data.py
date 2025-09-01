
#%%
from tqdm import tqdm
import pandas as pd
import src.pipeline_utils as pipeline_utils

def save_rows_to_csv(rows, filename="data/generated/disciplines.csv"):
    df = pd.DataFrame(rows, columns=["speciality_code", "speciality_name", "study_plan_url", "discipline_name", "work_program_url", "topics"])
    df.to_csv(filename, index=False, sep=';')

num_study_plans = 1
num_work_programs = 1

#%%
# extract bachelors disciplines: code xx.03.yy
speciality_df = pd.read_csv("data/download/specialities.csv", sep=';')
speciality_df = speciality_df[speciality_df['speciality_code'].str.strip().str.match(r'^\d{2}\.03\.\d{2}$', na=False)]
speciality_df # 187 rows

#%%
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

FLUSH_EVERY_SPECIALITIES = 3
FLUSH_MIN_ROWS = 200
NUM_WORKERS = 4

def log(msg: str, speciality_code=None, speciality_name=None):
    """Thread-safe log that prefixes the speciality if provided."""
    if speciality_code is not None and speciality_name is not None:
        tqdm.write(f"[{speciality_code} {speciality_name}] {msg}")
    else:
        tqdm.write(msg)

def process_speciality(speciality_code, speciality_name, num_study_plans, num_work_programs):
    log("START", speciality_code, speciality_name)
    local_rows = []

    try:
        study_plan_urls = pipeline_utils.get_study_plan_urls(speciality_code, speciality_name)
    except Exception as e:
        log(f"FAIL get_study_plan_urls: {e}", speciality_code, speciality_name)
        return local_rows

    used_study_plans = 0
    for study_plan_url in study_plan_urls:
        if used_study_plans >= num_study_plans:
            break

        plan_yielded = False
        try:
            discipline_names = pipeline_utils.extract_discipline_names(study_plan_url, speciality_name)
        except Exception as e:
            log(f"FAIL extract_discipline_names url={study_plan_url}: {e}", speciality_code, speciality_name)
            continue
        if not discipline_names or discipline_names == ['None']:
            log(f"No relevant disciplines in study_plan url={study_plan_url}", speciality_code, speciality_name)
            continue

        for discipline_name in discipline_names:
            used_work_programs = 0
            try:
                work_program_urls = pipeline_utils.get_work_program_urls(
                    discipline_name, speciality_code, speciality_name
                )
            except Exception as e:
                log(f"FAIL get_work_program_urls discipline='{discipline_name}': {e}",
                    speciality_code, speciality_name)
                work_program_urls = []

            for work_program_url in work_program_urls:
                if used_work_programs >= num_work_programs:
                    break
                try:
                    topics = pipeline_utils.extract_topics(work_program_url, discipline_name)
                except Exception as e:
                    log(f"FAIL extract_topics url={work_program_url} discipline='{discipline_name}': {e}",
                        speciality_code, speciality_name)
                    continue
                if not topics or topics == ['None']:
                    log(f"No topics url={work_program_url} discipline='{discipline_name}'",
                        speciality_code, speciality_name)
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
                plan_yielded = True

        if plan_yielded:
            used_study_plans += 1

    log(f"DONE → {len(local_rows)} rows", speciality_code, speciality_name)
    return local_rows


rows_buf, completed, total_written = [], 0, 0

def flush_rows():
    """Append and log how many rows were written."""
    global rows_buf, total_written
    if not rows_buf:
        return
    try:
        n = len(rows_buf)
        save_rows_to_csv(rows_buf)  # should append; header only if file doesn't exist
        total_written += n
        tqdm.write(f"[WRITE] +{n} rows (cumulative={total_written})")
    except Exception as e:
        tqdm.write(f"[WRITE-FAIL] Could not write {len(rows_buf)} rows: {e}")
    finally:
        rows_buf.clear()

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
            log(f"FAIL speciality task: {e}", scode, sname)
            batch = []
        if batch:
            rows_buf.extend(batch)

        completed += 1
        if completed % FLUSH_EVERY_SPECIALITIES == 0 or len(rows_buf) >= FLUSH_MIN_ROWS:
            flush_rows()

# final flush
flush_rows()
tqdm.write("[DONE] All specialities processed.")

# %%
