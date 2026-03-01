"""
counterfactual_agent_demo.py

Demo: Counterfactual Reasoning Agent – investigates how changing key
assumptions would alter outcomes. Counterfactual thinking is widely used
in causal inference, scientific explanation, risk analysis, and AI
safety (e.g., "If X had happened instead of Y, what would change?").

Flow:
1. Receive an event/claim and context
2. Agent uses LLM to identify pivotal factors
3. Generates alternative scenarios by altering those factors
4. Discusses how outcomes would differ and what this implies about
   causality or responsibility

This agent is popular for root-cause analysis, strategy, and
interpretability. It teaches models to think about "what if".

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python counterfactual_agent_demo.py

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


# ---------------------------------------------------------------------------
# LLM helpers (as usual)
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
# Counterfactual reasoning agent
# ---------------------------------------------------------------------------

class CounterfactualAgent:
    def __init__(self, llm_provider: str = "ollama", num_counterfactuals: int = 3):
        self.llm_provider = llm_provider
        self.num_counterfactuals = num_counterfactuals

    def identify_factors(self, event: str) -> List[str]:
        prompt = (
            "List the key causal factors or assumptions in the following event. "
            "Provide them as a numbered list.\n\n"
            f"Event: {event}\n\nFactors:"
        )
        response = call_llm(prompt, self.llm_provider)
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        factors = []
        for line in lines:
            parts = line.split(".",1)
            if len(parts) > 1:
                factors.append(parts[1].strip())
            else:
                factors.append(line)
        return factors

    def generate_counterfactuals(self, event: str, factors: List[str]) -> List[str]:
        prompt = (
            "Generate alternative versions of the event by changing one of the "
            "following causal factors at a time. Give each counterfactual a brief "
            "description.\n\nFactors: " + "; ".join(factors) + "\n\n"
            f"Original event: {event}\n\nCounterfactuals:"
        )
        response = call_llm(prompt, self.llm_provider)
        return [line.strip() for line in response.split("\n") if line.strip()]

    def analyze_counterfactuals(self, counterfactuals: List[str]) -> List[str]:
        analyses = []
        for cf in counterfactuals:
            prompt = (
                "For this counterfactual scenario, explain how the outcome would "
                "differ and what that implies about causality or responsibility.\n\n"
                f"Scenario: {cf}\n\nAnalysis:"
            )
            analyses.append(call_llm(prompt, self.llm_provider))
        return analyses

    def reason(self, event: str) -> Dict:
        factors = self.identify_factors(event)
        cfs = self.generate_counterfactuals(event, factors[: self.num_counterfactuals])
        analyses = self.analyze_counterfactuals(cfs)
        return {"factors": factors, "counterfactuals": cfs, "analyses": analyses}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = CounterfactualAgent(llm_provider="ollama")
    event = "The company launched a new product, but sales were disappointing."
    report = agent.reason(event)

    print("Identified factors:\n", report['factors'])
    print("\nCounterfactuals and analyses:")
    for cf, analysis in zip(report['counterfactuals'], report['analyses']):
        print(f"\nScenario: {cf}\nAnalysis: {analysis}\n")
