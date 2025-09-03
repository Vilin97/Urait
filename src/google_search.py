import requests
import json
from dotenv import load_dotenv
import os
import time

def parse_serper_response(response):
    """
    Parse Serper.dev search API response into a simplified list of results.
    Accepts requests.Response or a dict.
    Returns: List[Dict] with keys: position, title, url, snippet
    """
    # Normalize to dict
    try:
        data = response.json() if hasattr(response, "json") else (response or {})
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    organic = data.get("organic") or data.get("results") or []
    results = []
    for i, item in enumerate(organic, 1):
        results.append({
            "position": item.get("position", i),
            "title": item.get("title") or "",
            "url": item.get("link") or item.get("url") or "",
            "snippet": item.get("snippet") or item.get("description") or ""
        })
    return results

def _google_search(query):
    load_dotenv()
    api_key = os.getenv("SERPER_API_KEY")
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "location": "Moscow, Moscow, Russia",
        "gl": "ru",
        "hl": "ru",
        "autocorrect": False
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response

def search(query):
    time.sleep(0.1) # rate limiting
    response = _google_search(query)
    search_results = parse_serper_response(response)
    return search_results

