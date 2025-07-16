from ddgs import DDGS
from readability import Document
import requests
from typing import List

def search_web(query: str) -> List[str]:
    """Returns top 3 URLs from DuckDuckGo using ddgs."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query=query, max_results=3):
            if 'href' in r and r['href']:
                results.append(r['href'])
    return results

def scrape_text_from_url(url: str) -> str:
    """Fetches the URL content and extracts clean text using readability-lxml."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        doc = Document(response.text)
        return doc.summary()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching or processing URL {url}: {e}")
        return ""

def get_web_context(query: str) -> str:
    """Returns combined clean text from all URLs for the query."""
    urls = search_web(query)
    combined_text = []
    for url in urls:
        text = scrape_text_from_url(url)
        if text:
            combined_text.append(text)
    return "\n\n".join(combined_text)