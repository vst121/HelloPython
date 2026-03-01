"""
feedback_loop_agent_demo.py

Demo: Feedback-Loop Agent – continuously improves its responses via
explicit user feedback. After each answer, the user rates or corrects the
response; the agent incorporates that feedback to adjust prompt or
internal guidelines for future queries. This interactive refinement
process mimics RLHF and is a foundation of many deployed systems
(e.g. ChatGPT feedback upvotes, Microsoft Copilot improvements).

Workflow:
1. Agent answers user query.
2. User provides feedback (rating 1-5 or textual correction).
3. Agent updates a local "policy" prompt or correction memory.
4. Future answers are influenced by accumulated feedback.

This mechanism is extremely popular in customer-facing AI, knowledge
bases, and adaptive tutoring systems where personalization matters.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python feedback_loop_agent_demo.py

"""

import os
from typing import List, Dict

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


class FeedbackLoopAgent:
    """Agent that adapts based on user feedback."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        # store feedback history as list of dict
        self.feedback_memory: List[Dict] = []

    def generate_answer(self, question: str) -> str:
        base_prompt = (
            "You are a helpful assistant. Answer the user question clearly."
        )
        # incorporate feedback memory into prompt
        if self.feedback_memory:
            memory_text = "\n".join(
                [f"- {f['question']} -> feedback: {f['feedback']}" for f in self.feedback_memory]
            )
            base_prompt += (
                "\n\nThe following past feedback should inform your style and accuracy:\n" + memory_text
            )

        prompt = f"{base_prompt}\n\nQuestion: {question}\nAnswer:"
        return call_llm(prompt, self.llm_provider)

    def receive_feedback(self, question: str, answer: str, feedback: str):
        """Record feedback for a prior Q&A pair."""
        self.feedback_memory.append({
            "question": question,
            "answer": answer,
            "feedback": feedback,
        })
        print("Feedback recorded. Agent will use it for future queries.\n")

    def query(self, question: str) -> str:
        print(f"\nUser asks: {question}")
        answer = self.generate_answer(question)
        print(f"Agent answer:\n{answer}\n")
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
    agent = FeedbackLoopAgent(llm_provider="ollama")

    questions = [
        "What is the tallest mountain in the world?",
        "How does a solar eclipse happen?",
    ]

    for q in questions:
        ans = agent.query(q)
        # simulate user giving feedback
        if "mountain" in q:
            fb = "Correct but mention Everest is in the Himalayas."
        else:
            fb = "Good explanation, but add that it's due to moon alignment."
        agent.receive_feedback(q, ans, fb)

    # ask another question after feedback has been recorded
    agent.query("Tell me about Mount Everest.")
