"""
ethical_reasoning_agent_demo.py

Demo: Ethical Reasoning Agent – evaluates scenarios or decisions using
well-known ethical frameworks (utilitarianism, deontology, virtue
ethics, etc.). This type of agent has become popular in AI safety,
policy analysis, and conscience-like behavior (see OpenAI’s
Constitutional AI).

Flow:
1. Receive a scenario or question involving a moral dilemma
2. Agent prompts LLM to analyze the situation through multiple
   ethical lenses
3. Summarize trade-offs, possible actions, and provide a recommendation
4. Optionally generate a final decision and explanation

Popular because organizations need tools to surface ethical implications,
ensure fairness, and support transparent decision-making.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python ethical_reasoning_agent_demo.py

"""

import os
from typing import Dict

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
# LLM helpers (reuse existing pattern)
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


# ============================================================================
# Ethical Reasoning Agent
# ============================================================================

class EthicalReasoningAgent:
    """Analyzes dilemmas using multiple ethical frameworks."""

    FRAMEWORKS = [
        "Utilitarianism (maximize overall happiness)",
        "Deontology (duty-based ethics)",
        "Virtue Ethics (focus on character)",
        "Rights-based ethics (respect individual rights)",
        "Justice/Fairness (equitable outcomes)",
    ]

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider

    def analyze(self, scenario: str) -> str:
        prompt = (
            "You are an ethical reasoning assistant. Analyze the following "
            "scenario through multiple ethical frameworks and provide a "
            "summary of trade-offs and a recommended course of action.\n\n"
            f"Scenario: {scenario}\n\n"
            "Frameworks:\n"
        )
        for f in self.FRAMEWORKS:
            prompt += f"- {f}\n"
        prompt += "\nAnalysis:"  

        return call_llm(prompt, self.llm_provider)

    def decide(self, scenario: str) -> Dict[str, str]:
        analysis = self.analyze(scenario)
        decision_prompt = (
            "Based on the analysis above, what is the most ethical "
            "decision or recommendation? Provide in a concise statement.\n\n"
            f"Analysis:\n{analysis}\n\nDecision:"  
        )
        decision = call_llm(decision_prompt, self.llm_provider)
        return {"analysis": analysis, "decision": decision}


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    agent = EthicalReasoningAgent(llm_provider="ollama")
    scenario = (
        "A self-driving car must choose between swerving to avoid a group of "
        "pedestrians (crashing into a wall and possibly injuring its passenger) "
        "or staying its course and hitting the pedestrians."
    )
    result = agent.decide(scenario)

    print("Ethical analysis:\n")
    print(result["analysis"])
    print("\nRecommended decision:\n")
    print(result["decision"])
