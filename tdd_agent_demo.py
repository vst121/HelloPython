"""
tdd_agent_demo.py

Demo: Test-Driven Development (TDD) Agent.

This agent demonstrates a popular workflow where the LLM is instructed
 to write a unit test first, then produce code that satisfies the test,
 and finally run the test in a Python REPL tool to verify correctness.

The cycle:
1. RECEIVE specification or feature idea
2. GENERATE test code that describes expected behavior
3. WRITE implementation code to satisfy test
4. EXECUTE test in a sandbox environment (simulated via exec)
5. REFINE implementation until tests pass

TDD with LLMs has become widely adopted by developers as a way to
harness the model's ability to reason about requirements and ensure
correctness. Projects like GitHub Copilot and OpenAI encourage this
tool-assisted workflow.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python tdd_agent_demo.py

"""

import os
import json
import traceback
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


# ============================================================================
# Simple REPL tool simulation
# ============================================================================

def python_repl(code: str) -> Tuple[bool, str]:
    """Execute python code in a sandbox and return (success, output)."""
    try:
        # Redirect output capture
        local_vars = {}
        exec(code, {}, local_vars)
        return True, "Code executed successfully."
    except Exception as e:
        tb = traceback.format_exc()
        return False, tb


# ============================================================================
# TDD Agent
# ============================================================================

class TDD_Agent:
    """Test-driven development workflow orchestrator."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider

    def generate_test(self, feature_desc: str) -> str:
        """Ask LLM to produce a unit test stub based on feature description."""
        prompt = (
            f"# Feature: {feature_desc}\n"
            "# Write a Python unittest.TestCase class that tests this feature.\n"
            "# Do not implement the feature yet, only the test.\n"
        )
        return call_llm(prompt, self.llm_provider)

    def generate_code(self, test_code: str) -> str:
        """Ask LLM to write implementation that satisfies given test."""
        prompt = (
            "The following test is failing. Write Python code that makes it pass.\n"
            f"{test_code}\n"
            "# Provide only the implementation, not the test.\n"
        )
        return call_llm(prompt, self.llm_provider)

    def run_tdd_loop(self, feature_desc: str, max_iters: int = 3) -> Tuple[str, str]:
        """Perform TDD iterations: generate test, implement, run, refine."""
        test_code = self.generate_test(feature_desc)
        impl_code = """
# initial placeholder implementation
pass
"""
        result_log = ''

        for i in range(1, max_iters + 1):
            print(f"\n--- TDD iteration {i} ---")
            # implement based on current test
            impl_code = self.generate_code(test_code)
            print("Generated implementation:")
            print(impl_code)

            # combine test + implementation for execution
            combined = impl_code + "\n\n" + test_code + "\n"
            success, output = python_repl(combined)
            result_log += f"Iteration {i} success={success}\n{output}\n"

            if success:
                print("Tests passed!")
                break
            else:
                print("Tests failed, refining...")
                # ask LLM to fix the implementation with failure context
                prompt = (
                    "The previous implementation failed the test. Here is the failure:\n"
                    f"{output}\n"
                    "Update or correct the implementation to make the test pass.\n"
                    f"Current implementation:\n{impl_code}\n"
                )
                impl_code = call_llm(prompt, self.llm_provider)
        return test_code, impl_code


# ============================================================================
# LLM helpers (same as other demos)
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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1500)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("TDD Agent Demo")
    print("=" * 70)

    feature = "Create a function `add(a, b)` that returns the sum of two numbers."
    agent = TDD_Agent(llm_provider="ollama")
    test_code, impl_code = agent.run_tdd_loop(feature)

    print("\n--- Final test code ---")
    print(test_code)
    print("\n--- Final implementation ---")
    print(impl_code)
