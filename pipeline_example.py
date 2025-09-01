#%%
import pandas as pd
import json
import src.pipeline_utils as pipeline_utils

#%%
# Load a speciality from the list
speciality_df = pd.read_csv("data/download/specialities.csv", sep=';')
row = speciality_df.iloc[644] # choose just one for an example
speciality_code, speciality_name = row['speciality_code'], row['speciality_name']
print(f"Speciality: {speciality_code} {speciality_name}")

#%%
# Get URLs of study plans for a given speciality
study_plan_urls = pipeline_utils.get_study_plan_urls(speciality_code, speciality_name)
study_plan_urls
#%%
# Parse URL to get discipline names
study_plan_url = study_plan_urls[0] # choose just one for an example
print(f"Parsing study plan from {study_plan_url}")
discipline_names = pipeline_utils.extract_discipline_names(study_plan_url, speciality_name)
discipline_names

#%%
# Get URLs of work programs
discipline_name = discipline_names[0] # choose just one for an example
work_program_urls = pipeline_utils.get_work_program_urls(discipline_name, speciality_code, speciality_name)
work_program_urls

#%%
# Parse the work program to get topics
work_program_url = work_program_urls[0] # choose just one for an example
print(f"Discipline: {discipline_name}")
print(f"Parsing work program from {work_program_url}")
topics = pipeline_utils.extract_topics(work_program_url, discipline_name)
topics

#%%
rows = []

study_plan_urls = pipeline_utils.get_study_plan_urls(speciality_code, speciality_name)
for study_plan_url in study_plan_urls:
    try:
        discipline_names = pipeline_utils.extract_discipline_names(study_plan_url, speciality_name)
    except Exception as e:
        print(f"Failed to extract disciplines from {study_plan_url}: {e}")
        discipline_names = []
        continue

    if not discipline_names or discipline_names == ['None']:
        print(f"No relevant disciplines found in {study_plan_url}")
        continue

    for discipline_name in discipline_names:
        try:
            work_program_urls = pipeline_utils.get_work_program_urls(discipline_name, speciality_code, speciality_name)
        except Exception as e:
            print(f"Failed to get work program URLs for {discipline_name}: {e}")
            work_program_urls = []

        if not work_program_urls:
            continue

        for work_program_url in work_program_urls:
            try:
                topics = pipeline_utils.extract_topics(work_program_url, discipline_name)
            except Exception as e:
                print(f"Failed to extract topics from {work_program_url}: {e}")
                topics = []

            if not topics or topics == ['None']:
                continue

            rows.append({
                "speciality_code": speciality_code,
                "speciality_name": speciality_name,
                "study_plan_url": study_plan_url,
                "discipline_name": discipline_name,
                "work_program_url": work_program_url,
                "topics": "; ".join(topics)
            })

df = pd.DataFrame(rows, columns=["speciality_code", "speciality_name", "study_plan_url", "discipline_name", "work_program_url", "topics"])
output_filename = f"disciplines.csv"
df.to_csv(output_filename, index=False, sep=';')
print(f"Wrote {len(df)} rows to {output_filename}")