"""
scenario_planning_agent_demo.py

Demo: Scenario Planning Agent – generates and evaluates possible future
scenarios based on a given starting condition. Widely used in business
strategy, risk management, and creative brainstorming (e.g. McKinsey
scenario planning, Shell scenarios). The agent uses the LLM to propose
multiple plausible futures, assign rough likelihoods, and suggest
responses or mitigation strategies.

Flow:
1. User supplies a situation or trend (e.g., "AI adoption in finance")
2. Agent prompts LLM to generate several distinct scenarios with
   descriptions
3. Agent asks LLM to estimate probability and propose actions for each
4. Returns structured scenario report

Popular because organizations use it for resilience, innovation, and
‘what-if’ thinking. Advanced hybrid systems incorporate quantitative
models with qualitative scenarios.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python scenario_planning_agent_demo.py

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


class ScenarioPlanningAgent:
    """Agent that constructs future scenarios with probabilities and actions."""

    def __init__(self, llm_provider: str = "ollama", num_scenarios: int = 3):
        self.llm_provider = llm_provider
        self.num_scenarios = num_scenarios

    def generate_scenarios(self, prompt: str) -> List[Dict[str, str]]:
        system = (
            "You are a strategic foresight analyst. Given a prompt, "
            "generate several distinct future scenarios. For each scenario, "
            "provide a short title and description."
        )
        user = (
            f"Situation: {prompt}\n\n"
            f"Generate {self.num_scenarios} scenarios in the format:\n"
            "Title: ...\nDescription: ...\n"
        )
        raw = call_llm(user, self.llm_provider, system)
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        scenarios = []
        current = {}
        for line in lines:
            if line.startswith("Title:"):
                if current:
                    scenarios.append(current)
                current = {"title": line.split(":",1)[1].strip(), "description": ""}
            elif line.startswith("Description:"):
                current["description"] = line.split(":",1)[1].strip()
            else:
                if "description" in current:
                    current["description"] += " " + line
        if current:
            scenarios.append(current)
        return scenarios

    def evaluate_scenarios(self, scenarios: List[Dict[str, str]]) -> List[Dict[str, str]]:
        evaluated = []
        for scen in scenarios:
            prompt = (
                "For the following scenario, estimate its likelihood (low/medium/high) "
                "and suggest one key strategic response or mitigation.\n\n"
                f"Title: {scen['title']}\nDescription: {scen['description']}\n\n"
                "Answer with: Likelihood: ...\nResponse: ..."
            )
            out = call_llm(prompt, self.llm_provider)
            evaluated.append({"title": scen['title'], "description": scen['description'], "analysis": out})
        return evaluated

    def plan(self, situation: str) -> List[Dict[str, str]]:
        scenarios = self.generate_scenarios(situation)
        report = self.evaluate_scenarios(scenarios)
        return report


# ============================================================================
# LLM helpers (reuse pattern)
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
# Demo
# ============================================================================

if __name__ == "__main__":
    agent = ScenarioPlanningAgent(llm_provider="ollama", num_scenarios=3)
    situation = "Widespread adoption of autonomous vehicles by 2030"
    report = agent.plan(situation)
    for item in report:
        print(f"\nTitle: {item['title']}")
        print(f"Description: {item['description']}")
        print(f"Analysis:\n{item['analysis']}\n")
