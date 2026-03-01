"""
negotiation_agent_demo.py

Demo: Negotiation Agent – two LLM-powered agents negotiate a deal using
game-theoretic tactics. Each agent has a private utility function and
generates offers or counteroffers. Popular in research on multi-agent
emergent behavior, automated bargaining, and business process
automation (e.g., e-commerce price negotiation bots).

Flow:
1. Define negotiation subject (e.g., price of item) and each agent's
   utility curve
2. Agents alternate turns generating offers or accepting/rejecting
3. Agents use LLM reasoning to craft persuasive messages and estimate
   opponent utilities
4. Negotiation ends with agreement or deadline

This pattern is complex due to hidden information, strategy, and
natural-language interplay. It demonstrates advanced social reasoning
capabilities of LLMs.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python negotiation_agent_demo.py

"""

import os
import random
from typing import Dict, Any, Optional

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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Negotiator agent
# ---------------------------------------------------------------------------

class Negotiator:
    def __init__(self,
                 name: str,
                 pantry: Dict[str, Any],
                 utility: Dict[str, float],
                 llm_provider: str = "ollama"):
        self.name = name
        self.pantry = pantry  # hidden preferences
        self.utility = utility  # e.g., price -> satisfaction
        self.llm_provider = llm_provider
        self.history: list = []  # conversation history

    def make_offer(self, topic: str, last_offer: Optional[Dict[str, Any]] = None) -> str:
        prompt = (
            f"You are {self.name}, negotiating over {topic}. Your utility is {self.utility}. "
            "Given the previous offer (if any), propose a new offer or accept/reject. "
            "Explain your reasoning briefly.\n\n"
        )
        if last_offer:
            prompt += f"Last offer: {last_offer}\n\n"
        prompt += "Your response:"  
        response = call_llm(prompt, self.llm_provider)
        self.history.append((self.name, response))
        return response

    def interpret_response(self, response: str) -> Dict[str, Any]:
        # naive parse of number from text
        words = response.split()
        for w in words:
            try:
                val = float(w.strip("$,."))
                return {"price": val}
            except:
                continue
        return {}


# ---------------------------------------------------------------------------
# Negotiation workflow
# ---------------------------------------------------------------------------

def run_negotiation(topic: str,
                    agent1: Negotiator,
                    agent2: Negotiator,
                    max_turns: int = 10):
    last_offer = None
    for turn in range(max_turns):
        if turn % 2 == 0:
            speaker, listener = agent1, agent2
        else:
            speaker, listener = agent2, agent1

        resp = speaker.make_offer(topic, last_offer)
        print(f"{speaker.name}: {resp}")
        if "accept" in resp.lower():
            print(f"Agreement reached: {resp}")
            return
        last_offer = speaker.interpret_response(resp)
    print("No agreement reached within turn limit.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # define utilities: simple price satisfaction
    util1 = {"price": lambda p: -abs(p - 50)}  # wants 50
    util2 = {"price": lambda p: -abs(p - 70)}  # wants 70

    alice = Negotiator("Alice", pantry={}, utility=util1)
    bob = Negotiator("Bob", pantry={}, utility=util2)

    run_negotiation("price of vintage guitar", alice, bob, max_turns=6)
