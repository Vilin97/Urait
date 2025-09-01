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
NUM_WORKERS = 6           # tune for API limits
TOP_K = 10
OUT_CSV = "data/generated/discipline_course_suitability.csv"
FLUSH_EVERY = 1           # save after this many completed disciplines
# --------------------------------

# Logger
logger = logging.getLogger("suitability")
logger.setLevel(LOG_LEVEL)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setFormatter(fmt); fh.setLevel(LOG_LEVEL)
ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(LOG_LEVEL)
logger.addHandler(fh); logger.addHandler(ch)

# Ensure output dir
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

course_embeddings = np.vstack(courses_df['embedding'])

def process_discipline(i, row):
    speciality_name = row['speciality_name']
    discipline_name = row['discipline_name']
    discipline_topics = row['topics']
    embedding = row['embedding']

    logger.info(f"[{i}] Discipline: {discipline_name}")
    try:
        ids, scores = utils.get_most_similar(embedding, course_embeddings, top_k=TOP_K)
    except Exception as e:
        logger.error(f"[{i}] FAIL get_most_similar: {e}")
        return {"index": i, "top_courses": "[]", "is_covered": False, "coverage_ratio": 0.0}

    try:
        client = utils.get_gemini_client()
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
                "course_id": course_row['project_id'],
                "course_name": course_row['project_name'],
                "score": float(score),
                "answer": answer,
                "explanation": decision.get("explanation", ""),
                "covered_topics": decision.get("covered_topics", []),
                "missing_topics": decision.get("missing_topics", []),
                "ratio_covered_topics": ratio,
            })
            any_yes = any_yes or (answer == "Да")
            best_ratio = max(best_ratio, ratio)
        except Exception as e:
            logger.error(f"[{i}] FAIL per-course check: {e}")

    return {
        "index": i,
        "top_courses": json.dumps(results, ensure_ascii=False),
        "is_covered": any_yes,
        "coverage_ratio": best_ratio,
    }

# Prepare result columns to avoid SettingWithCopy surprises
results_df = disciplines_df.copy()
for col in ["top_courses", "is_covered", "coverage_ratio"]:
    if col not in results_df.columns:
        results_df[col] = np.nan

def flush_now():
    try:
        results_df.to_csv(OUT_CSV, index=False, sep=';')
        logger.info(f"[WRITE] Snapshot saved to {OUT_CSV}")
    except Exception as e:
        logger.error(f"[WRITE-FAIL] Could not save snapshot: {e}")

completed = 0
futures = []
try:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for i, row in results_df.iterrows():
            futures.append(ex.submit(process_discipline, i, row))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Suitability"):
            res = fut.result()  # if a worker crashed, log and continue
            i = res["index"]
            results_df.at[i, 'top_courses'] = res["top_courses"]
            results_df.at[i, 'is_covered'] = res["is_covered"]
            results_df.at[i, 'coverage_ratio'] = res["coverage_ratio"]

            completed += 1
            if completed % FLUSH_EVERY == 0:
                flush_now()
finally:
    # final flush (also runs on exceptions / KeyboardInterrupt)
    flush_now()
    logger.info(f"[DONE] Processed {completed} disciplines total.")
