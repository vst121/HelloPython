"""
react_agent_demo.py

Demo: ReAct (Reasoning + Acting) Agent - A very popular LLM agent pattern.

This agent:
1. THINKS: Reasons about the problem and decides what to do next
2. ACTS: Calls tools/functions (calculator, web search, code execution, etc.)
3. OBSERVES: Analyzes the result and decides next step or concludes
4. Loops until the task is solved

Usage:
- pip install requests
- Optionally: set OLLAMA_HOST, OLLAMA_MODEL, OPENAI_API_KEY
- python react_agent_demo.py

This pattern is used by Claude, ChatGPT with plugins, and other leading agents.
"""

import os
import json
import re
from typing import Optional, Any

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
# Tool Definitions (Simulated)
# ============================================================================

TOOLS = {
    "calculator": {
        "name": "calculator",
        "description": "Performs basic math: add, subtract, multiply, divide",
        "usage": "calculator(operation, a, b) → result",
    },
    "web_search": {
        "name": "web_search",
        "description": "Searches the web for information (simulated)",
        "usage": "web_search(query) → [list of results]",
    },
    "python_repl": {
        "name": "python_repl",
        "description": "Executes Python code and returns output",
        "usage": "python_repl(code) → output",
    },
    "knowledge_base": {
        "name": "knowledge_base",
        "description": "Query a knowledge base for facts",
        "usage": "knowledge_base(topic) → facts",
    },
}

TOOL_DESCRIPTIONS = "\n".join(
    [f"- {t['name']}: {t['description']}" for t in TOOLS.values()]
)


def calculator(operation: str, a: float, b: float) -> float:
    """Simulate a calculator tool."""
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }
    result = ops.get(operation.lower(), lambda x, y: None)(a, b)
    return result


def web_search(query: str) -> list:
    """Simulate web search results."""
    # Mock results for demo
    mock_results = {
        "python": [
            "Python is a high-level programming language",
            "Python is widely used in AI and data science",
            "Python has a large ecosystem of libraries",
        ],
        "ai": [
            "AI is transforming industries",
            "LLMs are a breakthrough in AI",
            "ReAct agents are popular in LLM research",
        ],
        "weather": [
            "Today's weather in New York: Sunny, 72°F",
            "Tomorrow: Chance of rain, 65°F",
        ],
    }

    for key, results in mock_results.items():
        if key.lower() in query.lower():
            return results

    return ["No results found for query: " + query]


def python_repl(code: str) -> str:
    """Simulate Python execution (safe subset)."""
    try:
        # In production, use RestrictedPython or similar
        # For demo, we'll just handle basic math expressions
        if code.count("import") > 0:
            return "Error: imports not allowed in demo"

        result = eval(code)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def knowledge_base(topic: str) -> str:
    """Query a knowledge base."""
    facts = {
        "python": "Python was created by Guido van Rossum in 1991. It's known for readability and simplicity.",
        "agents": "AI agents are systems that perceive and act on their environment. ReAct is a popular reasoning pattern.",
        "tools": "Tools enable agents to interact with external systems: calculators, web APIs, databases, code execution.",
    }

    for key, fact in facts.items():
        if key.lower() in topic.lower():
            return fact

    return f"No facts found for topic: {topic}"


def execute_tool(tool_name: str, *args, **kwargs) -> str:
    """Execute a tool by name and return result as string."""
    if tool_name == "calculator":
        operation, a, b = args[0], args[1], args[2]
        result = calculator(operation, a, b)
        return f"{a} {operation} {b} = {result}"
    elif tool_name == "web_search":
        results = web_search(args[0])
        return "\n".join([f"  {i+1}. {r}" for i, r in enumerate(results)])
    elif tool_name == "python_repl":
        return python_repl(args[0])
    elif tool_name == "knowledge_base":
        return knowledge_base(args[0])
    else:
        return f"Unknown tool: {tool_name}"


# ============================================================================
# LLM Integration
# ============================================================================

def call_llm(prompt: str, provider: str = "ollama", system_prompt: str = None) -> str:
    """Call LLM with a prompt."""
    if provider.lower() == "openai":
        return call_openai(prompt, system_prompt)
    else:
        return call_ollama(prompt, system_prompt)


def call_ollama(prompt: str, system_prompt: str = None) -> str:
    """Call Ollama model."""
    if not OLLAMA_AVAILABLE:
        return "(Ollama not available - install requests)"

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3")
    url = f"{host.rstrip('/')}/chat?model={model}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            url,
            json={"messages": messages},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                return msg.get("content", "").strip()

        return str(data)
    except Exception as e:
        return f"(Ollama error: {e})"


def call_openai(prompt: str, system_prompt: str = None) -> str:
    """Call OpenAI model."""
    if not OPENAI_AVAILABLE:
        return "(OpenAI not available)"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(No OpenAI API key)"

    try:
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error: {e})"


# ============================================================================
# ReAct Agent
# ============================================================================

class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent.
    Loops through: Think → Act → Observe → Repeat until done.
    """

    def __init__(self, llm_provider: str = "ollama", max_steps: int = 10):
        self.llm_provider = llm_provider
        self.max_steps = max_steps
        self.history = []
        self.step_count = 0

    def think(self, task: str, context: str = "") -> str:
        """Agent thinks about the task and decides next action."""
        system_prompt = f"""You are a ReAct agent. Your job is to solve problems by:
1. THINKING: Reason about what needs to be done
2. ACTING: Choose a tool and use it, or provide the final answer
3. OBSERVING: Analyze the result

Available tools:
{TOOL_DESCRIPTIONS}

When using a tool, format it as: [TOOL_NAME](args)
Example: [calculator](add, 5, 3) or [web_search](python) or [python_repl](2**3)

When you have the answer, format it as: [FINAL_ANSWER](answer)"""

        prompt = f"Task: {task}\n\nContext: {context}" if context else f"Task: {task}"

        response = call_llm(prompt, self.llm_provider, system_prompt)
        return response

    def act(self, response: str) -> tuple[str, str]:
        """Parse agent response and execute tool if needed."""
        # Look for [TOOL_NAME](args) pattern
        tool_match = re.search(r"\[(\w+)\]\(([^)]+)\)", response)

        if tool_match:
            tool_name = tool_match.group(1).lower()
            args_str = tool_match.group(2)

            # Simple arg parsing (could be more sophisticated)
            args = [arg.strip().strip("'\"") for arg in args_str.split(",")]

            if tool_name in TOOLS:
                result = execute_tool(tool_name, *args)
                return tool_name, result
            else:
                return "unknown", f"Unknown tool: {tool_name}"

        # Check for final answer
        if "[FINAL_ANSWER]" in response:
            answer_match = re.search(r"\[FINAL_ANSWER\]\(([^)]+)\)", response)
            if answer_match:
                return "final", answer_match.group(1)

        return "none", "No action detected"

    def run(self, task: str) -> str:
        """Run the agent loop until task is solved or max steps reached."""
        print(f"\n{'='*70}")
        print(f"ReAct Agent (using {self.llm_provider})")
        print(f"{'='*70}")
        print(f"\nTask: {task}\n")

        context = ""
        self.step_count = 0

        while self.step_count < self.max_steps:
            self.step_count += 1
            print(f"--- Step {self.step_count} ---")

            # THINK
            print("Thinking...")
            thought = self.think(task, context)
            print(f"Agent: {thought[:200]}...")

            # ACT
            print("Acting...")
            tool_used, result = self.act(thought)
            print(f"Tool: {tool_used}")
            print(f"Result: {result}\n")

            # Check if done
            if tool_used == "final":
                print(f"{'='*70}")
                print(f"Final Answer: {result}")
                print(f"{'='*70}")
                return result

            # OBSERVE - add to context for next iteration
            context += f"\nStep {self.step_count}: Used {tool_used}, got: {result}"

            self.history.append(
                {
                    "step": self.step_count,
                    "thought": thought,
                    "action": tool_used,
                    "observation": result,
                }
            )

            if self.step_count >= self.max_steps:
                print(f"Max steps ({self.max_steps}) reached.")
                break

        return "Agent failed to reach conclusion."


# ============================================================================
# Main - Demo Tasks
# ============================================================================

if __name__ == "__main__":
    # Example tasks
    tasks = [
        "What is 12 times 5?",
        "Tell me about Python and how it's used in AI",
        "Calculate 100 divided by 4, then multiply by 2",
    ]

    agent = ReActAgent(llm_provider="ollama")

    # Run first task
    print("\nRunning demo with mock tools (no LLM calls needed)...\n")

    # For demo, we'll show the pattern without waiting for LLM
    print("NOTE: This demo uses mock LLM responses. ")
    print("To use a real LLM:")
    print("  - Install Ollama (phi3) or set OPENAI_API_KEY")
    print("  - The agent will then reason and use tools interactively\n")

    # Show agent flow
    task = tasks[0]
    print(f"Demo Task: {task}")
    print("\n--- Agent Workflow (Example) ---")
    print("Step 1: [THINK] I need to multiply 12 by 5")
    tool_result = execute_tool("calculator", "multiply", 12, 5)
    print(f"Step 2: [ACT] {tool_result}")
    print("Step 3: [OBSERVE] The answer is 60")
    print("[FINAL_ANSWER](60)")
    print("\n✓ Agent successfully completed the task!\n")

    # Uncomment to run with real LLM (requires Ollama running)
    # result = agent.run(tasks[0])
