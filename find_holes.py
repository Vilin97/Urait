"""For each discipline, find top k most similar courses by embedding similarity, and determine suitability."""

#%%
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import src.utils as utils

#%%
# load disciplines with embeddings
disciplines_df = pd.read_csv('data/generated/disciplines_with_embeddings.csv', sep=';')
disciplines_df['embedding'] = disciplines_df['embedding'].apply(lambda s: np.array(json.loads(s), dtype=np.float32))
disciplines_df

#%%
# load courses with embeddings
courses_df = pd.read_csv('data/generated/courses.csv')
courses_df['embedding'] = courses_df['embedding'].apply(lambda s: np.array(json.loads(s), dtype=np.float32))
courses_df

#%%
import logging, json, os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ------------ Config ------------
LOG_FILE = "suitability.log"
LOG_LEVEL = logging.INFO
NUM_WORKERS = 5           # tune for API limits
TOP_K = 10
OUT_CSV = "data/generated/discipline_course_suitability.csv"
# --------------------------------

# Logger (tqdm-friendly console + file)
logger = logging.getLogger("suitability")
logger.setLevel(LOG_LEVEL)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(fmt)
fh.setLevel(LOG_LEVEL)
logger.addHandler(fh)

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

ch = TqdmLoggingHandler()
ch.setFormatter(fmt)
ch.setLevel(LOG_LEVEL)
logger.addHandler(ch)

# Ensure output dir
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# Precompute embeddings once
course_embeddings = np.vstack(courses_df['embedding'])

def process_discipline(i, row):
    """Process one discipline; returns dict with index + results."""
    speciality_name = row['speciality_name']
    discipline_name = row['discipline_name']
    discipline_topics = row['topics']
    embedding = row['embedding']

    logger.info(f"[{i}] [START] {discipline_name}, num_topics={len(discipline_topics.split(';'))}")
    try:
        ids, scores = utils.get_most_similar(embedding, course_embeddings, top_k=TOP_K)
    except Exception as e:
        logger.error(f"[{i}] FAIL get_most_similar: {e}")
        return {"index": i, "top_courses": "[]", "is_covered": False, "coverage_ratio": 0.0}

    try:
        client = utils.get_gemini_client()  # create per-thread unless known thread-safe
    except Exception as e:
        logger.error(f"[{i}] FAIL get_gemini_client: {e}")
        return {"index": i, "top_courses": "[]", "is_covered": False, "coverage_ratio": 0.0}

    results, best_ratio, any_yes = [], 0.0, False
    for idx, score in zip(ids[:TOP_K], scores[:TOP_K]):
        try:
            course_row = courses_df.iloc[idx]
            decision = utils.determine_course_suitability(
                speciality_name, discipline_name, discipline_topics,
                course_row['project_name'], course_row['topics'], client
            )
            answer = decision.get("answer", "Неизвестно")
            ratio = float(decision.get("ratio_covered_topics", 0.0))
            results.append({
                "course_id": int(course_row['project_id']),  # force to Python int
                "course_name": str(course_row['project_name']),
                "score": float(score),
                "answer": str(answer),
                "explanation": str(decision.get("explanation", "")),
                "covered_topics": str(decision.get("covered_topics", '')),
                "missing_topics": str(decision.get("missing_topics", '')),
                "ratio_covered_topics": float(ratio),
            })

            any_yes = any_yes or (answer == "Да")
            best_ratio = max(best_ratio, ratio)
            logger.info(f"    [{i}] [COMPARE] score={score:.4f} | answer={answer+(' ' if answer=='Да' else '')} | ratio={ratio:.2f} | {course_row['project_name']}")
            if any_yes:
                break  # stop early if any suitable course is found
        except Exception as e:
            logger.error(f"[{i}] FAIL per-course check: {e}")

    logger.info(f"[{i}] [DONE] {discipline_name} | any suitable: {any_yes}, best ratio: {best_ratio:.2f}")
    return {
        "index": i,
        "top_courses": json.dumps(results, ensure_ascii=False),
        "is_covered": any_yes,
        "coverage_ratio": best_ratio,
    }

# Prepare result columns (stable defaults)
results_df = disciplines_df.copy()
if "top_courses" not in results_df.columns: results_df["top_courses"] = ""
if "is_covered"  not in results_df.columns: results_df["is_covered"]  = False
if "coverage_ratio" not in results_df.columns: results_df["coverage_ratio"] = 0.0

def flush_now(idx_list):
    try:
        to_save = results_df.loc[idx_list].drop(columns=['embedding'])
        header = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
        to_save.to_csv(OUT_CSV, index=False, sep=';', mode='a', header=header)
        logger.info(f"{idx_list} [WRITE] Appended {len(idx_list)} rows to {OUT_CSV}")
    except Exception as e:
        logger.error(f"[WRITE-FAIL] Could not append rows: {e}")

completed = 0
futures = []
try:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for i, row in results_df.iterrows():
            futures.append(ex.submit(process_discipline, i, row))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Suitability"):
            try:
                res = fut.result()
            except Exception as e:
                logger.error(f"[FUTURE-FAIL] [{i}] Discipline {row['discipline_name']} task crashed: {e}")
                completed += 1
                flush_now([i])
                continue

            i = res["index"]
            results_df.at[i, 'top_courses'] = res["top_courses"]
            results_df.at[i, 'is_covered'] = res["is_covered"]
            results_df.at[i, 'coverage_ratio'] = res["coverage_ratio"]

            completed += 1
            flush_now([i])
finally:
    logger.info(f"[DONE] Processed {completed} disciplines total.")
