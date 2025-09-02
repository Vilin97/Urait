import numpy as np
from google import genai
from google.genai import types
import httpx
import os
from dotenv import load_dotenv
import json

DEFAULT_MODEL = "gemini-2.5-flash"

def get_gemini_client(api_key_name="GOOGLE_API_KEY"):
    """Make a genai client from an env var.
    
    TODO: switch to litellm for model interchangeability."""
    load_dotenv()
    api_key = os.getenv(api_key_name)
    client = genai.Client(api_key=api_key)
    return client

### Document parsing utilities ###
def _generate_from_url(url, prompt, mime_type, client, model=DEFAULT_MODEL):
    """Helper to fetch URL content and generate response from it."""
    doc_data = httpx.get(url).content
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=doc_data, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text

def parse_pdf(url, prompt, client, model=DEFAULT_MODEL):
    return _generate_from_url(url, prompt, mime_type="application/pdf", client=client, model=model)

def parse_html(url, prompt, client, model=DEFAULT_MODEL):
    return _generate_from_url(url, prompt, mime_type="text/html", client=client, model=model)

def parse_document(url, prompt, client, model=DEFAULT_MODEL):
    """Parse a document from a URL using the given prompt."""
    if url.endswith(".pdf"):
        return parse_pdf(url, prompt, client, model=model)
    elif url.endswith(".html"):
        return parse_html(url, prompt, client, model=model)
    else:
        raise ValueError("Unsupported document type. Only PDF and HTML are supported.")

### Embedding-related utilities ###
def embed_text(text, client, model="gemini-embedding-001", output_dimensionality=768):
    """Embed text using the specified embedding model."""
    response = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=output_dimensionality))
    embedding = np.array(response.embeddings[0].values, dtype=float)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def load_course_embeddings(npz_path="course_embeddings/course_embeddings.npz"):
    """Load course embeddings from disk into a dict of id -> embedding.
    
    TODO: switch to a better storage format, maybe just store in the same CSV as courses."""
    with np.load(npz_path) as data:
        loaded_ids = data["ids"].astype(int)
        vecs = data["embeddings"].astype(float)

    embeddings_by_id = {int(i): vec for i, vec in zip(loaded_ids, vecs)}
    return embeddings_by_id

def get_most_similar(embedding, embeddings, top_k=5):
    """Get the top_k most similar course ids to the given embedding.
    embeddings: np.ndarray of shape (num_courses, embedding_dim)
    Returns: List of indices of the most similar courses and their similarity scores."""
    sims = embeddings @ embedding
    top_indices = np.argsort(sims)[-top_k:][::-1]
    top_scores = sims[top_indices]
    return top_indices, top_scores
    

### Course-discipline suitability determination ###
SCHEMA = {
    "type": "object",
    "propertyOrdering": ["covered_topics","missing_topics","explanation","answer"],
    "required": ["covered_topics","missing_topics","explanation","answer"],
    "properties": {
        "covered_topics": {"type": "string"},
        "missing_topics": {"type": "string"},
        "explanation": {"type": "string"},
        "answer": {"type": "string", "enum": ["Да","Нет"]},
    },
}

def determine_course_suitability(speciality, discipline_name, discipline_topics, course_name, course_topics, client, model=DEFAULT_MODEL):
    """Determine if a course is suitable for teaching a discipline based on topics."""

    main_prompt = """You are a Russian expert educational consultant specializing in higher education (university/college).
    Your task is to determine whether a given course is suitable for teaching a specific college discipline.
    You are given the speciality/major, discipline name and the topics of the discipline, as well as the textbook title and topics.
    You must decide if the course is suitable for teaching the discipline, and also determine the topics of the discipline that are covered by the course, and the topics that are missing. One of these lists may be empty.
    Do not invent topics. Only use the topics listed in 'Discipline Topics'.
    Use your expert judgment for semantic equivalence of topics (paraphrases, synonyms, abbreviations, close variants). Prefer meaning over exact wording.
    The explanation must be in Russian, no longer than 2 sentences.
    Respond ONLY with a single valid JSON object (no Markdown, no comments, no extra text).
    'answer' is 'Да' if the course covers >70% core topics of the discipline; otherwise 'Нет'.
    """

    schema = f"""
    Schema:
    {{
    "covered_topics": ["<list of topics from discipline covered by the course, separated by `;`>"],
    "missing_topics": ["<list of topics from discipline NOT covered by the course, separated by `;`>"],
    "explanation": "<short explanation in Russian>",
    "answer": <"Да" | "Нет">
    }}

    Speciality: {speciality}
    Discipline: {discipline_name}
    Discipline Topics: {discipline_topics}

    Course Title: {course_name}
    Course Topics: {course_topics}
    """

    prompt = main_prompt + schema

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": SCHEMA,
        },
    )

    try:
        text = response.text.strip()
        start = text.find('{')
        end = text.rfind('}')
        parsed = json.loads(text[start:end+1])
        num_covered = len(parsed.get("covered_topics", "").split(';'))
        num_missing = len(parsed.get("missing_topics", "").split(';'))
        ratio_covered = num_covered / (num_covered + num_missing) if (num_covered + num_missing) > 0 else 0.0
        parsed['ratio_covered_topics'] = ratio_covered
    except (json.JSONDecodeError, ValueError) as e:
        print("Failed to parse JSON:", e)
        parsed = {"answer": "Ошибка", "explanation": "Не удалось распарсить ответ модели.", "covered_topics": [], "missing_topics": []}
    return parsed