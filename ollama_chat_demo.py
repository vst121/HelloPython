"""
ollama_chat_demo.py

Demo: Chat with a local Ollama model (e.g., phi3).

Usage:
- pip install requests
- Ensure Ollama is running locally (default: http://localhost:11434)
- Optionally set env vars: OLLAMA_HOST, OLLAMA_MODEL
- python ollama_chat_demo.py

This script sends a simple chat payload to Ollama's `/chat` endpoint
and prints the assistant response. Adjust `messages` for multi-turn usage.
"""

import os
import json

try:
    import requests
except ImportError:
    raise SystemExit("Please install requests: pip install requests")

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "phi3")


def chat(messages, model=MODEL, host=DEFAULT_HOST, timeout=60):
    """Send `messages` to Ollama `/chat?model=...` and return a best-effort text.

    messages: list of dicts like [{"role":"system","content":"..."}, ...]
    """
    url = f"{host.rstrip('/')}/chat?model={model}"
    payload = {"messages": messages}
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # Try to parse JSON and extract likely assistant text
    try:
        data = resp.json()
    except ValueError:
        return resp.text

    # Common Ollama response patterns vary; try several fallbacks
    if isinstance(data, dict):
        # 1) choices -> message/delta/content
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            parts = []
            for c in choices:
                # c may contain 'message' or 'delta' or 'content' or 'text'
                if isinstance(c, dict):
                    msg = c.get("message") or c.get("delta") or {}
                    if isinstance(msg, dict):
                        content = msg.get("content") or msg.get("text")
                    else:
                        content = c.get("content") or c.get("text")
                else:
                    content = None
                if content:
                    parts.append(content)
            if parts:
                return "\n".join(parts)
        # 2) top-level 'text' key
        if "text" in data and isinstance(data["text"], str):
            return data["text"]

    # Fallback: pretty-print JSON
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short haiku about programming."},
    ]

    print(f"Sending to Ollama at {DEFAULT_HOST} using model '{MODEL}'...\n")
    try:
        out = chat(messages)
        print("Response:\n")
        print(out)
    except requests.RequestException as e:
        print("Request error:", e)
        print("Ensure Ollama is running locally, e.g.: http://localhost:11434")
