#%%
import src.google_search as google_search
import src.utils as utils
import pandas as pd

#%%
# Load a speciality from the list
speciality_df = pd.read_csv("specialities.csv", sep=';')
row = speciality_df.iloc[644] # choose just one for an example
speciality_code, speciality_name = row['speciality_code'], row['speciality_name']
print(f"Speciality: {speciality_code} {speciality_name}")

#%%
# Get URLs of study plans for a given speciality
def get_study_plan_urls(speciality_code, speciality_name):
    query = f"Направление подготовки {speciality_code} {speciality_name} \"учебный план\" pdf"
    search_results = google_search.search(query)
    study_plan_urls = [r.get('url') for r in search_results if r.get('title').startswith('[PDF]') and r.get('url').endswith('.pdf')]
    return study_plan_urls

study_plan_urls = get_study_plan_urls(speciality_code, speciality_name)
study_plan_urls
#%%
# Parse URL to get discipline names
def extract_discipline_names(study_plan_url, speciality_name):
    prompt = f"""Extract the names of all disciplines that are directly related to {speciality_name} and its closely connected applications. This includes both required and elective courses. 

    Do not include:
    - Disciplines that are clearly outside the main subject area (for example, if the subject is history, drop math/physics/programming; if the subject is mathematics, drop languages, law, history, etc.).
    - Disciplines that are purely administrative or non-academic in nature, e.g. 'Научно-исследовательская работа'
    - Seminars, labs, practicals, internships, and other non-lecture courses.
    - Broad categories or headings, e.g. 'Математика' or 'Физика'

    Return the discipline names in Russian, separated by semicolon `;`.

    If you cannot find any relevant disciplines, return `None`.
    """
    llm_client = utils.get_gemini_client()
    parsed = utils.parse_document(study_plan_url, prompt, llm_client)
    discipline_names = parsed.split('; ')
    return discipline_names

study_plan_url = study_plan_urls[0] # choose just one for an example
print(f"Parsing study plan from {study_plan_url}")
discipline_names = extract_discipline_names(study_plan_url, speciality_name)
discipline_names

#%%
# Get URLs of work programs
def get_work_program_urls(discipline_name, speciality_code, speciality_name):
    query = f"\"{discipline_name}\" рабочая программа дисциплины {speciality_code} {speciality_name} pdf"
    search_results = google_search.search(query)
    work_program_urls = [r.get('url') for r in search_results if r.get('title').startswith('[PDF]') and r.get('url').endswith('.pdf')]
    return work_program_urls

discipline_name = discipline_names[0] # choose just one for an example
work_program_urls = get_work_program_urls(discipline_name, speciality_code, speciality_name)
work_program_urls

#%%
# Parse the work program to get topics
def extract_topics(work_program_url, discipline_name):
    prompt = f"""Extract all the topics covered in course {discipline_name}, all in Russian. Only include academic topics, not administrative. 
    Respond only with the names of topics, separated by semicolon `;`.
    If you cannot find any academic topics, return `None`."""

    llm_client = utils.get_gemini_client()
    parsed = utils.parse_document(work_program_url, prompt, llm_client)
    topics = parsed.split('; ')
    return topics

work_program_url = work_program_urls[0] # choose just one for an example
print(f"Discipline: {discipline_name}")
print(f"Parsing work program from {work_program_url}")
topics = extract_topics(work_program_url, discipline_name)
topics
