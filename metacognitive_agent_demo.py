"""
metacognitive_agent_demo.py

Demo: Metacognitive Agent – an LLM that not only reasons but also
monitors its reasoning process for mistakes, gaps, or biases. The agent
explicitly reflects on its own answers, identifies uncertainties, and
can revise or ask for clarification. This meta-level thinking is
increasingly popular in research (e.g., GPT-4 Turbo’s system messages
promoting self-reflection) and improves reliability in critical domains.

Flow:
1. Agent receives question
2. Generates preliminary answer with chain-of-thought
3. Performs self-audit: checks for reasoning errors, missing steps
4. If issues found, revises answer or asks follow-up question
5. Returns final answer plus reflection log

Applications: safety-critical assistance, scientific reasoning,
law/legal analysis, medical diagnosis.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python metacognitive_agent_demo.py

"""

import os
from typing import Tuple

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


class MetacognitiveAgent:
    """Agent that reasons and reflects on its reasoning."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider

    def initial_answer(self, question: str) -> str:
        prompt = (
            "Answer the question below. Show your chain-of-thought reasoning "
            "step by step in a numbered list.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        return call_llm(prompt, self.llm_provider)

    def self_audit(self, answer: str) -> Tuple[str, bool]:
        prompt = (
            "Review the answer below for logical errors, gaps in reasoning, "
            "unsupported claims, or biases. If you find any issue, explain it "
            "and suggest a correction. If the answer is sound, say 'no issues'.\n\n"
            f"Answer with reasoning:\n{answer}\n\nAudit:"
        )
        audit = call_llm(prompt, self.llm_provider)
        issues_found = "no issues" not in audit.lower()
        return audit, issues_found

    def revise_answer(self, question: str, audit: str) -> str:
        prompt = (
            "Based on the audit below, revise the original answer to correct "
            "any problems. Provide a final polished response.\n\n"
            f"Audit:\n{audit}\n\nQuestion: {question}\n\nRevised Answer:"
        )
        return call_llm(prompt, self.llm_provider)

    def query(self, question: str) -> Tuple[str, str, str]:
        print(f"\nIncoming question: {question}")
        initial = self.initial_answer(question)
        print(f"\nInitial response:\n{initial}\n")

        audit, issues = self.self_audit(initial)
        print(f"Audit log:\n{audit}\n")

        if issues:
            final = self.revise_answer(question, audit)
            print(f"Revised answer:\n{final}\n")
        else:
            final = initial
            print("No revisions needed.\n")

        return initial, audit, final


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
    agent = MetacognitiveAgent(llm_provider="ollama")
    questions = [
        "What factors contributed to the fall of the Roman Empire?",
        "How does photosynthesis convert sunlight into energy?",
    ]

    for q in questions:
        agent.query(q)
