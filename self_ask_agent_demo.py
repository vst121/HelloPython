"""
self_ask_agent_demo.py

Demo: Self-Ask Agent (Decompose-with-Subquestions Pattern).

Workflow:
1. RECEIVE: Complex user question
2. SELF-ASK: Agent asks simpler subquestions to itself via LLM
3. GATHER: Collect answers to each subquestion
4. SYNTHESIZE: Combine subanswers into final response

This strategy helps LLMs reason step-by-step by breaking problems
into smaller queries. It has become popular following the "self-ask"
paper by Google Research (2022) and is widely adopted in QA systems.

Use cases:
- Multi-hop question answering
- Open-domain search with reasoning
- Complex instruction comprehension

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python self_ask_agent_demo.py

"""

import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass

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


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SubQuestion:
    question: str
    answer: str = ""


@dataclass
class SelfAskResult:
    original_question: str
    subquestions: List[SubQuestion]
    final_answer: str


# ============================================================================
# Agent
# ============================================================================

class SelfAskAgent:
    """Implements the self-ask reasoning pattern."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider

    def generate_subquestions(self, question: str) -> List[SubQuestion]:
        """Ask the LLM to decompose into simpler questions."""
        system_prompt = (
            "You are a reasoning assistant. Break down the user's complex "
            "question into a sequence of simpler subquestions that can be "
            "answered independently."
        )

        prompt = (
            f"User question: {question}\n\n"
            "Generate a numbered list of subquestions."
        )

        response = call_llm(prompt, self.llm_provider, system_prompt)
        subs: List[SubQuestion] = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # remove leading numbers
            q = line.lstrip("0123456789.- ")
            subs.append(SubQuestion(question=q))

        return subs

    def answer_subquestions(self, subs: List[SubQuestion]) -> None:
        """Answer each subquestion."""
        for sub in subs:
            prompt = f"Answer briefly: {sub.question}"
            sub.answer = call_llm(prompt, self.llm_provider)

    def synthesize_final_answer(self, question: str, subs: List[SubQuestion]) -> str:
        """Combine subanswers into final response."""
        text = "\n".join(f"- {s.question} -> {s.answer}" for s in subs)
        prompt = (
            f"Original question: {question}\n\n"
            f"Subquestion answers:\n{text}\n\n"
            "Using the above information, provide a comprehensive answer to the original question."
        )
        return call_llm(prompt, self.llm_provider)

    def query(self, question: str) -> SelfAskResult:
        print(f"\n[SELF-ASK] Starting with question: {question}")
        subs = self.generate_subquestions(question)
        print(f"Generated {len(subs)} subquestions")
        for idx, s in enumerate(subs, 1):
            print(f"  {idx}. {s.question}")

        self.answer_subquestions(subs)
        for idx, s in enumerate(subs, 1):
            print(f"  ✓ {idx}. {s.answer}")

        final = self.synthesize_final_answer(question, subs)
        print(f"Final answer: {final}\n")

        return SelfAskResult(original_question=question, subquestions=subs, final_answer=final)


# ============================================================================
# LLM integration
# ============================================================================

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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("Self-Ask Agent Demo")
    print("=" * 70)

    complex_questions = [
        "What is the capital of Australia and what are three famous landmarks there?",
        "Explain how a solar eclipse occurs and why it doesn't happen every month.",
        "Who wrote 'To Kill a Mockingbird' and what inspired the story?",
    ]

    agent = SelfAskAgent(llm_provider="ollama")

    for q in complex_questions:
        agent.query(q)

    # Uncomment to ask your own question:
    # custom = input("Enter a complex question: ")
    # agent.query(custom)
