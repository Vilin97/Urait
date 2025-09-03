import src.google_search as google_search
import src.utils as utils

def get_study_plan_urls(speciality_code, speciality_name, university_name):
    """Get URLs of study plans for a given speciality"""
    query = f"Направление подготовки {speciality_code} {speciality_name} \"учебный план\" {university_name} pdf"
    search_results = google_search.search(query)
    study_plan_urls = [r.get('url') for r in search_results]
    return study_plan_urls

def extract_discipline_names(study_plan_url, speciality_name):
    """Parse URL to get discipline names"""
    prompt = f"""Extract the names of all disciplines that are directly related to {speciality_name} and its closely connected applications from this study plan (учебный план). This includes both required and elective courses. 

    Do not include:
    - Disciplines that are clearly outside the main subject area (for example, if the subject is history, drop math/physics/programming; if the subject is mathematics, drop languages, law, history, etc.).
    - Disciplines that are purely administrative or non-academic in nature, e.g. 'Научно-исследовательская работа'
    - Seminars, labs, practicals, internships, and other non-lecture courses.
    - Broad categories or headings, e.g. 'Математика' or 'Физика'

    Return the discipline names in Russian, separated by semicolon `;`. Do not include any other text in your response.

    If you cannot find any relevant disciplines (for example, the webpage is clearly not a study plan or is an error webpage), return `None`.
    """
    llm_client = utils.get_gemini_client()
    parsed = utils.parse_document(study_plan_url, prompt, llm_client)
    discipline_names = parsed.split('; ')
    return discipline_names

def get_work_program_urls(discipline_name, speciality_code, speciality_name, university_name):
    """Get URLs of work programs"""
    query = f"\"{discipline_name}\" рабочая программа дисциплины {speciality_code} {speciality_name} {university_name} pdf"
    search_results = google_search.search(query)
    work_program_urls = [r.get('url') for r in search_results]
    return work_program_urls

def extract_topics(work_program_url, discipline_name):
    """Parse the work program to get topics"""
    prompt = f"""Extract all the topics covered in course {discipline_name}, all in Russian. Only include academic topics, not administrative. 
    Respond only with the names of topics, separated by semicolon `;`.
    If you cannot find any academic topics, return `None`."""

    llm_client = utils.get_gemini_client()
    parsed = utils.parse_document(work_program_url, prompt, llm_client)
    topics = parsed.split('; ')
    return topics