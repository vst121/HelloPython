"""
scientific_agent_demo.py

Demo: Scientific Research Agent – an advanced LLM-based agent that
acts like a junior researcher. It receives a high-level research
question, formulates a hypothesis, designs and executes simple
experiments (simulated via Python code), analyzes results, and iterates
toward a conclusion. This workflow mimics autonomous science agents
that plan, experiment, and learn from data (e.g., AlphaCode for
research, DARPA’s ASIST).

Flow:
1. User supplies a research question (e.g., "Do larger samples reduce
   variance?")
2. Agent uses LLM to propose a hypothesis and experimental design
3. The agent writes Python code to run the experiment (simulation)
4. Executes the code, gathers numeric results
5. Summarizes findings with the LLM and either refines experiment or
   draws conclusions
6. Optionally repeats several iterations

This pattern is very complex and advanced, integrating planning,
code synthesis/execution, data analysis, and iterative refinement.

Usage:
- pip install requests numpy
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python scientific_agent_demo.py

"""

import os
import textwrap
import traceback
import numpy as np
from typing import Dict, Any

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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1500)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Scientific Agent
# ---------------------------------------------------------------------------

class ScientificAgent:
    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.history: Dict[str, Any] = {}

    def propose_study(self, question: str) -> Dict[str, str]:
        """Use LLM to propose hypothesis and experimental design."""
        prompt = (
            "You are a research assistant. Given a high-level question, "
            "propose a clear hypothesis and describe a simple experiment in Python "
            "that could test it (using numpy for simulation). Respond with a JSON "
            "object containing 'hypothesis' and 'code'.\n\nQuestion: " + question
        )
        response = call_llm(prompt, self.llm_provider)
        try:
            proposal = json.loads(response)
        except Exception:
            # fallback: simple parse
            proposal = {"hypothesis": "", "code": response}
        return proposal

    def run_experiment(self, code: str) -> Dict[str, Any]:
        """Execute provided Python code and capture results."""
        local_vars = {}
        try:
            exec(code, {"np": np}, local_vars)
            return {"success": True, "results": local_vars.get("results", None)}
        except Exception as e:
            return {"success": False, "error": traceback.format_exc()}

    def analyze_results(self, question: str, hypothesis: str, results: Any) -> str:
        prompt = (
            "You are a scientific analyst. Given the question, hypothesis, "
            "and experiment results (a Python variable `results`), explain what "
            "the results imply and whether they support the hypothesis.\n\n"
            f"Question: {question}\nHypothesis: {hypothesis}\nResults: {results}\n\nAnalysis:"
        )
        return call_llm(prompt, self.llm_provider)

    def investigate(self, question: str, iterations: int = 2) -> None:
        for i in range(iterations):
            print(f"\n--- iteration {i+1} ---")
            study = self.propose_study(question)
            hypothesis = study.get("hypothesis", "")
            code = study.get("code", "")
            print(f"Hypothesis: {hypothesis}")
            print(f"Running code:\n{code}\n")
            outcome = self.run_experiment(code)
            if not outcome["success"]:
                print("Experiment failed:\n", outcome["error"])
                break
            results = outcome["results"]
            print("Experiment results:", results)
            analysis = self.analyze_results(question, hypothesis, results)
            print("Analysis:\n", analysis)
            question = question + " (follow up)"  # simulate refining


# ---------------------------------------------------------------------------
# Demo script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    agent = ScientificAgent(llm_provider="ollama")
    research_question = "Does increasing sample size reduce the variance of the sample mean?"
    agent.investigate(research_question, iterations=2)
