"""
context_manager_agent_demo.py

Demo: Context Window Manager Agent – maintains long conversations by
summarizing past interactions to stay within the model's context limit.

Flow:
1. Track full conversation in a history list
2. When history grows large, generate a concise summary of earlier
   messages using the LLM itself
3. Replace detailed history with summary to free up tokens
4. Continue chatting while retaining key information

Used widely in systems that handle multi-hour chats, customer support,
and knowledge workflows where the context window is limited (GPT-4
8K/32K tokens). Very popular as modal architectures and memory services
rely on it.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python context_manager_agent_demo.py

"""

import os
from typing import List

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


class ContextManagerAgent:
    """Agent that summarizes history to manage limited context."""

    def __init__(self, llm_provider: str = "ollama", max_history: int = 10):
        self.llm_provider = llm_provider
        self.max_history = max_history
        self.history: List[str] = []  # chat log lines

    def add_user_message(self, text: str):
        self.history.append(f"User: {text}")
        self._shrink_history_if_needed()

    def add_agent_message(self, text: str):
        self.history.append(f"Agent: {text}")
        self._shrink_history_if_needed()

    def _shrink_history_if_needed(self):
        if len(self.history) > self.max_history:
            # summarize earliest half of history
            to_summarize = "\n".join(self.history[: len(self.history) // 2])
            summary = self._summarize_text(to_summarize)
            # replace those lines with summary line
            self.history = [f"[Summary of earlier conversation: {summary}]"
                             ] + self.history[len(self.history) // 2 :]

    def _summarize_text(self, text: str) -> str:
        prompt = (
            "Summarize the following conversation succinctly, preserving key points:\n"
            f"{text}\n\nSummary:" 
        )
        return call_llm(prompt, self.llm_provider)

    def generate_response(self, user_input: str) -> str:
        self.add_user_message(user_input)
        prompt = "\n".join(self.history) + f"\nAgent:"  
        response = call_llm(prompt, self.llm_provider)
        self.add_agent_message(response)
        return response

    def get_history(self) -> str:
        return "\n".join(self.history)


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
    print("Context Window Manager Agent Demo")
    print("=" * 60)

    agent = ContextManagerAgent(llm_provider="ollama", max_history=6)
    interactions = [
        "Hello, can you help me?",
        "What's the weather like in Paris?",
        "Tell me a joke.",
        "Remind me what we talked about earlier.",
        "Now explain quantum mechanics in simple terms.",
        "And finally, how do I cook pasta?",
        "Thanks! Any final tips?",
    ]

    for msg in interactions:
        print(f"\nUser: {msg}")
        res = agent.generate_response(msg)
        print(f"Agent: {res}\n")
        print("Current history:")
        print(agent.get_history())
        print("---")
