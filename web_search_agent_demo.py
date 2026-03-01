"""
web_search_agent_demo.py

Demo: Web Search Augmented Agent – combines an LLM with a search tool
for up-to-date information. This pattern powers Bing Chat, Google Bard,
and many enterprise assistants that need real-time facts.

Flow:
1. RECEIVE: User asks question
2. DECIDE: Agent determines whether to query the web
3. SEARCH: Simulated search API returns top-k snippets
4. SYNTHESIZE: LLM uses search results to craft a final response

Search results are mocked for demo; replace `mock_search_api` with a
real API (SerpAPI, Google, Bing, etc.).

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python web_search_agent_demo.py

"""

import os
import random
from typing import List, Dict, Any

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = False
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mock search API
# ---------------------------------------------------------------------------

def mock_search_api(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """Simulate a web search returning snippets."""
    samples = [
        {
            "title": f"Result {i} for {query}",
            "snippet": f"This is a mock snippet about '{query}', item {i}.",
            "url": f"http://example.com/{query.replace(' ', '_')}/{i}",
        }
        for i in range(1, num_results + 1)
    ]
    random.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class WebSearchAgent:
    """Agent that augments LLM responses using an external search tool."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider

    def should_search(self, query: str) -> bool:
        """Decide whether to use search. Simple heuristic: if query mentions 'latest' or '2024'."""
        keywords = ["latest", "recent", "2024", "2025", "current"]
        return any(k in query.lower() for k in keywords)

    def search(self, query: str) -> List[Dict[str, Any]]:
        print("🔍 Performing web search...")
        results = mock_search_api(query)
        for res in results:
            print(f"  • {res['title']}: {res['snippet']}")
        return results

    def synthesize_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        prompt = (
            "You are an assistant that can use web search results to answer questions."
            " Use the snippets to support your answer and cite URLs when appropriate.\n\n"
            f"Question: {query}\n\n"
            "Search results:\n"
        )
        for r in search_results:
            prompt += f"- {r['title']}: {r['snippet']} ({r['url']})\n"
        prompt += "\nProvide a comprehensive answer that incorporates this information."

        return call_llm(prompt, self.llm_provider)

    def query(self, question: str) -> str:
        print(f"\nUser question: {question}")
        if self.should_search(question):
            results = self.search(question)
            answer = self.synthesize_answer(question, results)
        else:
            answer = call_llm(question, self.llm_provider)
        print(f"\nFinal answer:\n{answer}\n")
        return answer


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(prompt: str, provider: str = "ollama", system_prompt: str = None) -> str:
    if provider.lower() == "openai":
        return call_openai(prompt, system_prompt)
    else:
        return call_ollama(prompt, system_prompt)


def call_ollama(prompt: str, system_prompt: str = None) -> str:
    if not OLLAMA_AVAILABLE:
        return "(Ollama not available)"
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3")
    url = f"{host.rstrip('/')}/chat?model={model}"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(url, json={"messages": messages}, headers={"Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                return msg.get("content", "").strip()
        return str(data)
    except Exception as e:
        return f"(Error: {e})"


def call_openai(prompt: str, system_prompt: str = None) -> str:
    if not OPENAI_AVAILABLE:
        return "(OpenAI not available)"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(No API key)"
    try:
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1500)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = WebSearchAgent(llm_provider="ollama")

    queries = [
        "What is the capital of France?",
        "What are the latest updates on Python 3.12 release?",
        "Explain how photosynthesis works.",
    ]

    for q in queries:
        agent.query(q)
