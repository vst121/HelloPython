"""
data_analysis_agent_demo.py

Demo: Data Analysis Agent – combines LLM reasoning with actual data
analytics using pandas. This pattern is popular in business intelligence,
notebook assistants, and data scientist workflows (e.g., ChatGPT with
Python, Copilot Labs). The agent reads a dataset, computes statistics,
and uses the LLM to interpret and summarize findings.

Flow:
1. Load a CSV or DataFrame
2. Compute descriptive statistics and simple plots (here text only)
3. Prompt LLM with results to generate insights
4. Optionally answer follow-up questions based on data

Requires pandas for basic computation. The LLM helps translate numeric
output into natural-language insights.

Usage:
- pip install requests pandas
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python data_analysis_agent_demo.py

"""

import os
import json
import pandas as pd
from typing import Any

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
# LLM integration (same as other demos)
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
# Data Analysis Agent
# ============================================================================

class DataAnalysisAgent:
    """Agent that loads data and uses LLM to generate insights."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.df: pd.DataFrame = pd.DataFrame()

    def load_sample_data(self):
        # create a simple sample dataset in memory
        data = {
            "employee": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "salary": [120000, 80000, 115000, 95000, 105000],
            "department": ["Eng", "HR", "Eng", "Sales", "Eng"],
            "years": [5, 3, 7, 2, 4],
        }
        self.df = pd.DataFrame(data)

    def compute_statistics(self) -> str:
        desc = self.df.describe(include="all").to_string()
        counts = self.df['department'].value_counts().to_string()
        return desc + "\n\nDepartment counts:\n" + counts

    def ask_question(self, question: str) -> str:
        stats = self.compute_statistics()
        prompt = (
            "You are a data analyst. Use the dataset statistics below "
            "to answer the user question in natural language.\n\n"
            f"Dataset statistics:\n{stats}\n\nQuestion: {question}\n\nAnswer:"
        )
        return call_llm(prompt, self.llm_provider)


# ============================================================================
# Demo usage
# ============================================================================

if __name__ == "__main__":
    agent = DataAnalysisAgent(llm_provider="ollama")
    agent.load_sample_data()

    print("Sample data loaded:")
    print(agent.df)
    print("\nStatistics:")
    print(agent.compute_statistics())

    questions = [
        "Which department has the highest average salary?",
        "What is the typical experience level of employees?",
    ]

    for q in questions:
        print(f"\nUser question: {q}")
        answer = agent.ask_question(q)
        print(f"Agent insight:\n{answer}\n")
