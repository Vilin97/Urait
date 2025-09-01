import numpy as np
from google import genai
from google.genai import types
import httpx
import os
from dotenv import load_dotenv
import json

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SUITABILITY_PROMPT = """Given the course title and topics, determine if this course can be used to teach the given discipline in a college course. The course might contain many topics, but as long as it covers most of the topics (>70%) of the discipline, it is acceptable. Respond in Russian with a short explanation."""

def get_gemini_client(api_key_name="GOOGLE_API_KEY"):
    """Make a genai client from an env var.
    
    TODO: switch to litellm for model interchangeability."""
    load_dotenv()
    api_key = os.getenv(api_key_name)
    client = genai.Client(api_key=api_key)
    return client

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

def get_most_similar(embedding, embeddings_by_id, top_k=5):
    """Get the top_k most similar course ids to the given embedding."""
    ids = list(embeddings_by_id.keys())
    vecs = np.vstack([embeddings_by_id[i] for i in ids])
    sims = vecs @ embedding
    top_indices = np.argsort(sims)[-top_k:][::-1]
    top_ids = [ids[i] for i in top_indices]
    top_sims = [sims[i] for i in top_indices]
    return list(zip(top_ids, top_sims))

def determine_course_suitability(discipline_name, discipline_topics, course_name, course_topics, client, model=DEFAULT_MODEL, main_prompt=DEFAULT_SUITABILITY_PROMPT):
    """Determine if a course is suitable for teaching a discipline based on topics."""

    prompt = main_prompt+f"""\nRespond ONLY with a single valid JSON object (no extra text). Schema:
    {{
    "explanation": "<short explanation in Russian>",
    "answer": "Да" or "Нет",
    }}

    Discipline: {discipline_name}
    Discipline Topics: {discipline_topics}

    Course Title: {course_name}
    Topics: {course_topics}
    """

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
    )

    try:
        text = response.text.strip()
        start = text.find('{')
        end = text.rfind('}')
        parsed = json.loads(text[start:end+1])
    except (json.JSONDecodeError, ValueError) as e:
        print("Failed to parse JSON:", e)
        parsed = {"answer": "Ошибка", "explanation": "Не удалось распарсить ответ модели."}
    return parsed.get('answer'), parsed.get('explanation')