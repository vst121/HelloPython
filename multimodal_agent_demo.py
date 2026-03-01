"""
multimodal_agent_demo.py

Demo: Multimodal Agent – handles both text and image inputs. Popularized
by models like GPT-4V, LLaVA, and other vision-language systems. This
demo shows how an agent might receive a text prompt plus an image path,
interpret the image (via a mock vision API), and produce a combined
response.

Flow:
1. Receive user query with optional image path
2. If image provided, call vision API to get a description
3. Compose prompt including image description and text
4. Call LLM to generate multimodal-aware answer

In practice, the vision API might be an external service (OpenAI image
model, BLIP, etc.). Here we mock it to keep dependencies minimal.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python multimodal_agent_demo.py

"""

import os
from typing import Optional

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Vision component (mock)
# ---------------------------------------------------------------------------

def mock_vision_api(image_path: str) -> str:
    """Simulate performing image understanding. In real use, this would
call a vision model and return a caption or features."""
    # very simplistic placeholder logic
    base = os.path.basename(image_path).lower()
    if "cat" in base:
        return "A photo of a cat sitting on a windowsill."
    elif "chart" in base:
        return "A bar chart showing sales by quarter."
    elif "dog" in base:
        return "A dog playing in the park."
    else:
        return "A generic image (details not provided)."


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class MultimodalAgent:
    """Agent that can incorporate image descriptions into its responses."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider

    def analyze(self, text: str, image_path: Optional[str] = None) -> str:
        image_description = None
        if image_path:
            print(f"Analyzing image at {image_path}...")
            image_description = mock_vision_api(image_path)
            print(f"Image description: {image_description}\n")

        prompt = """You are a multimodal assistant. """
        if image_description:
            prompt += f"Use the following image information when answering.\n\nImage description: {image_description}\n\n"
        prompt += f"User query: {text}\n\nProvide a helpful response."

        return call_llm(prompt, self.llm_provider)


# ---------------------------------------------------------------------------
# LLM helpers (same pattern as other demos)
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
                return choices[0].get("message", {}).get("content", "").strip()
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
    agent = MultimodalAgent(llm_provider="ollama")

    queries = [
        ("Describe what you see and its significance.", "cat_picture.jpg"),
        ("What might this chart indicate about our business?", "sales_chart.png"),
        ("Tell me a joke.", None),
    ]

    for text, image in queries:
        answer = agent.analyze(text, image)
        print(f"Agent response:\n{answer}\n")
