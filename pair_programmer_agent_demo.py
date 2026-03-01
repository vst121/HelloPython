"""
pair_programmer_agent_demo.py

Demo: Pair Programmer Agent – simulates two LLMs collaborating as a
pair-programming team (driver and navigator). The driver writes code
while the navigator reviews, suggests improvements, and asks clarifying
questions. This back-and-forth greatly improves code quality and is a
popular workflow with tools like GitHub Copilot, ChatGPT, and Anthropic's
Code Assistant.

Workflow:
1. DRIVER receives task and writes initial code
2. NAVIGATOR critiques the code, points out bugs, and suggests tests
3. DRIVER updates implementation
4. Repeat until satisfied

This pattern encourages iterative refinement, peer review, and
continuous feedback – mimicking human pair programming.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python pair_programmer_agent_demo.py

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


# ---------------------------------------------------------------------------
# LLM utilities
# ---------------------------------------------------------------------------

def call_llm(prompt: str, provider: str = "ollama", system: str = None) -> str:
    if provider.lower() == "openai":
        return call_openai(prompt, system)
    else:
        return call_ollama(prompt, system)


def call_ollama(prompt: str, system: str = None) -> str:
    if not OLLAMA_AVAILABLE:
        return "(Ollama unavailable)"
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3")
    url = f"{host.rstrip('/')}/chat?model={model}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
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


def call_openai(prompt: str, system: str = None) -> str:
    if not OPENAI_AVAILABLE:
        return "(OpenAI unavailable)"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(No API key)"
    try:
        client = OpenAI(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1500)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Pair programming roles
# ---------------------------------------------------------------------------

def driver_step(task_description: str, previous_code: str = "") -> str:
    system = "You are the driver. Write or update Python code based on the task and any previous code."
    prompt = f"Task: {task_description}\n\nPrevious code:\n{previous_code}\n\nProvide the updated code snippet."  
    return call_llm(prompt, "ollama", system)


def navigator_review(code: str) -> str:
    system = "You are the navigator. Review the provided code snippet for bugs, missing test cases, style, and correctness. Suggest improvements or ask questions."
    prompt = f"Here is the code to review:\n{code}\n\nProvide feedback and any test ideas."
    return call_llm(prompt, "ollama", system)


def navigator_tests(code: str) -> str:
    system = "As the navigator, write one or more simple pytest-style unit tests that validate the behavior of the code."
    prompt = f"Code to test:\n{code}\n\nWrite tests."  
    return call_llm(prompt, "ollama", system)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def pair_program(task: str, iterations: int = 3) -> Tuple[str, str]:
    code = """# start with an empty file"""
    log = ""
    for i in range(1, iterations + 1):
        log += f"\n=== Iteration {i} ===\n"
        # driver writes code
        code = driver_step(task, code)
        log += "Driver wrote code:\n" + code + "\n"

        # navigator reviews
        review = navigator_review(code)
        log += "Navigator review:\n" + review + "\n"

        # navigator provides tests
        tests = navigator_tests(code)
        log += "Navigator tests:\n" + tests + "\n"

        # driver might incorporate suggestions (simple re-run for demo)
        code = driver_step(task + " (incorporate feedback)", code)
        log += "Driver updated code after feedback:\n" + code + "\n"

    return code, log


# ---------------------------------------------------------------------------
# Demo execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Pair Programmer Agent Demo")
    print("=" * 60)

    task = "Implement a function `factorial(n)` that returns the factorial of n."
    final_code, conversation_log = pair_program(task, iterations=2)

    print("\nFinal code produced by driver:")
    print(final_code)
    print("\nConversation log:")
    print(conversation_log)
