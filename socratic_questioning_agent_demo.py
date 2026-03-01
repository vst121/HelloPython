"""
socratic_questioning_agent_demo.py

Demo: Socratic Questioning Agent – an LLM that answers by iteratively
asking itself clarifying questions to deepen understanding.

Pattern:
1. Receive user question
2. Ask a clarifying/subquestion about key aspects
3. Answer the subquestion
4. Repeat until the agent has enough information to provide a
   comprehensive solution
5. Synthesize final answer incorporating the dialogue

Inspired by the Socratic method and used in systems aiming for
explainability, transparency, and improved reasoning. Popular in
educational chatbots and critical thinking tools.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python socratic_questioning_agent_demo.py

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


class SocraticAgent:
    """Agent that uses self-questioning to build answers."""

    def __init__(self, llm_provider: str = "ollama", max_rounds: int = 3):
        self.llm_provider = llm_provider
        self.max_rounds = max_rounds
        self.dialogue: List[str] = []  # stores question/answer pairs

    def ask_clarifying(self, question: str) -> str:
        """Generate a clarifying subquestion."""
        prompt = (
            "You are an inquisitive assistant. Given a question, propose "
            "a follow-up question that would help clarify or deepen the answer."
            f"\n\nOriginal question: {question}\n\n"
            "Provide a single, concise subquestion."
        )
        return call_llm(prompt, self.llm_provider)

    def answer(self, prompt_text: str) -> str:
        """Answer a prompt (either original or subquestion)."""
        return call_llm(prompt_text, self.llm_provider)

    def run(self, question: str) -> str:
        self.dialogue.append(f"User: {question}")
        current_question = question

        for i in range(self.max_rounds):
            subq = self.ask_clarifying(current_question)
            if not subq or subq.lower().startswith("no"):
                break
            self.dialogue.append(f"Agent (clarify): {subq}")
            ans = self.answer(subq)
            self.dialogue.append(f"Agent answer: {ans}")
            current_question = subq  # drill down

        # after rounds, answer original question with amassed context
        synthesis_prompt = (
            "Based on the conversation below, provide a final, comprehensive "
            "answer to the original user question.\n\n" + "\n".join(self.dialogue)
        )
        final = self.answer(synthesis_prompt)
        self.dialogue.append(f"Final answer: {final}")
        return final


# ---------------------------------------------------------------------------
# LLM calls
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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Socratic Questioning Agent Demo")
    print("=" * 70)

    agent = SocraticAgent(llm_provider="ollama", max_rounds=2)
    questions = [
        "Why is the sky blue?",
        "How can I improve my sleep habits?",
        "What causes inflation in economics?",
    ]

    for q in questions:
        print(f"\n---\nOriginal: {q}")
        answer = agent.run(q)
        print(f"Answer:\n{answer}\n")

    print("Conversation log:")
    for line in agent.dialogue:
        print(line)
