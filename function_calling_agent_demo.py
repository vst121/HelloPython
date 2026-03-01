"""
function_calling_agent_demo.py

Demo: Function Calling Agent (Tool Use Pattern).

This agent:
1. UNDERSTANDS: Receives a query and interprets intent
2. PLANS: Decides which tools/functions to call
3. CALLS: Executes tools with LLM-generated arguments
4. OBSERVES: Processes tool results
5. RESPONDS: Generates final answer based on tool outputs

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python function_calling_agent_demo.py

This pattern powers ChatGPT plugins, Claude tool use, OpenAI function calling,
and is THE most widely used agent pattern in production LLM systems today.
Features: Deterministic tool selection, argument inference, result integration.
"""

import os
import json
import re
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass

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
# Tool Definitions & Registry
# ============================================================================

@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class ToolDefinition:
    """Formal definition of a tool/function the agent can call."""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format (like OpenAI function calling)."""
        params = {}
        required = []
        
        for param in self.parameters:
            params[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required,
            },
        }


class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        """Register a tool."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)

    def call_tool(self, name: str, **kwargs) -> str:
        """Call a tool by name with arguments."""
        tool = self.get_tool(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        try:
            result = tool.function(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error calling {name}: {e}"

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas."""
        return [tool.to_schema() for tool in self.tools.values()]

    def get_descriptions(self) -> str:
        """Get human-readable tool descriptions."""
        descriptions = "Available Tools:\n"
        for tool in self.tools.values():
            descriptions += f"- {tool.name}: {tool.description}\n"
            for param in tool.parameters:
                required = "(required)" if param.required else "(optional)"
                descriptions += f"  • {param.name} ({param.type}): {param.description} {required}\n"
        return descriptions


# ============================================================================
# Sample Tools
# ============================================================================

# Mathematical operations
def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# Information tools
def get_weather(city: str) -> str:
    """Simulate getting weather data."""
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 58°F",
        "tokyo": "Clear, 68°F",
        "paris": "Rainy, 62°F",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


def search_knowledge(query: str) -> str:
    """Simulate searching a knowledge base."""
    knowledge = {
        "python": "Python is a high-level, interpreted programming language known for simplicity and readability.",
        "ai": "AI (Artificial Intelligence) is the simulation of human intelligence by machines.",
        "agent": "An agent is an autonomous system that perceives its environment and takes actions to achieve goals.",
    }
    
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    
    return f"No information found about '{query}'"


def code_formatter(code: str, language: str = "python") -> str:
    """Simulate code formatting."""
    return f"Formatted {language} code (demo):\n{code[:100]}..."


def create_tool_registry() -> ToolRegistry:
    """Create and populate tool registry."""
    registry = ToolRegistry()

    # Math tools
    registry.register(ToolDefinition(
        name="add",
        description="Add two numbers",
        parameters=[
            ToolParameter("a", "number", "First number", required=True),
            ToolParameter("b", "number", "Second number", required=True),
        ],
        function=add,
    ))

    registry.register(ToolDefinition(
        name="subtract",
        description="Subtract two numbers",
        parameters=[
            ToolParameter("a", "number", "First number", required=True),
            ToolParameter("b", "number", "Second number", required=True),
        ],
        function=subtract,
    ))

    registry.register(ToolDefinition(
        name="multiply",
        description="Multiply two numbers",
        parameters=[
            ToolParameter("a", "number", "First number", required=True),
            ToolParameter("b", "number", "Second number", required=True),
        ],
        function=multiply,
    ))

    registry.register(ToolDefinition(
        name="divide",
        description="Divide two numbers",
        parameters=[
            ToolParameter("a", "number", "Numerator", required=True),
            ToolParameter("b", "number", "Denominator", required=True),
        ],
        function=divide,
    ))

    # Information tools
    registry.register(ToolDefinition(
        name="get_weather",
        description="Get current weather for a city",
        parameters=[
            ToolParameter("city", "string", "City name", required=True),
        ],
        function=get_weather,
    ))

    registry.register(ToolDefinition(
        name="search_knowledge",
        description="Search a knowledge base for information",
        parameters=[
            ToolParameter("query", "string", "Search query", required=True),
        ],
        function=search_knowledge,
    ))

    registry.register(ToolDefinition(
        name="code_formatter",
        description="Format code in a specific language",
        parameters=[
            ToolParameter("code", "string", "Code to format", required=True),
            ToolParameter("language", "string", "Programming language", required=False),
        ],
        function=code_formatter,
    ))

    return registry


# ============================================================================
# Function Calling Agent
# ============================================================================

class FunctionCallingAgent:
    """
    Function Calling Agent using the tool use pattern.
    This is THE most popular agent pattern in modern LLM systems.
    Powers: ChatGPT plugins, Claude tool use, OpenAI function calling.
    """

    def __init__(self, tool_registry: ToolRegistry, llm_provider: str = "ollama"):
        self.tools = tool_registry
        self.llm_provider = llm_provider
        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []

    def generate_tool_prompt(self, user_query: str) -> str:
        """Generate prompt for tool selection."""
        tool_descriptions = self.tools.get_descriptions()
        
        prompt = f"""You are an AI agent with access to the following tools:

{tool_descriptions}

User request: {user_query}

Based on the user request, decide which tool(s) to use and call them with appropriate arguments.
Format your response as:
[TOOL_NAME](arg1=value1, arg2=value2)

You can call multiple tools. After calling tools, provide a final answer based on the results.
If no tools are needed, just answer directly."""
        
        return prompt

    def parse_tool_calls(self, response: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Parse tool calls from agent response."""
        # Pattern: [TOOL_NAME](args)
        pattern = r'\[(\w+)\]\(([^)]+)\)'
        matches = re.findall(pattern, response)
        
        tool_calls = []
        for tool_name, args_str in matches:
            # Parse arguments
            args = {}
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    args[key.strip()] = value.strip().strip('"\'')
            
            tool_calls.append((tool_name, args))
        
        return tool_calls

    def process_query(self, query: str, use_llm: bool = True) -> Dict[str, Any]:
        """Process a user query using function calling."""
        print(f"\n{'='*70}")
        print(f"Processing: {query}")
        print(f"{'='*70}\n")

        # Step 1: Get LLM to decide which tools to use
        prompt = self.generate_tool_prompt(query)
        
        if use_llm:
            print("🤖 Asking LLM which tools to use...")
            response = call_llm(prompt, self.llm_provider)
        else:
            # For demo, simulate response
            response = "[search_knowledge](query=agent)"
            print("📝 Demo response (simulated):")

        print(f"LLM Response: {response[:200]}...\n")

        # Step 2: Parse tool calls
        print("🔍 Parsing tool calls...")
        tool_calls = self.parse_tool_calls(response)
        
        if not tool_calls:
            print("No tools to call, providing direct response.\n")
            return {
                "query": query,
                "tool_calls": [],
                "answer": response,
            }

        # Step 3: Execute tools
        print(f"Found {len(tool_calls)} tool call(s):\n")
        results = {}

        for tool_name, args in tool_calls:
            print(f"  ▶ Calling {tool_name}({args})...")
            result = self.tools.call_tool(tool_name, **args)
            print(f"    Result: {result}")
            results[tool_name] = result

        # Step 4: Generate final answer
        print("\n💭 Generating final answer...")
        final_prompt = f"""Based on the tool results:
{json.dumps(results, indent=2)}

Original question: {query}

Provide a clear, concise answer incorporating the tool results."""

        final_answer = call_llm(
            final_prompt,
            self.llm_provider,
            "You are a helpful assistant that answers questions using tool results."
        )

        print(f"\n✓ Final Answer:\n{final_answer}\n")

        return {
            "query": query,
            "tool_calls": [{"name": name, "args": args} for name, args in tool_calls],
            "tool_results": results,
            "answer": final_answer,
        }

    def run_interactive_demo(self):
        """Run interactive demo with sample queries."""
        print(f"\n{'='*70}")
        print(f"Function Calling Agent Demo (using {self.llm_provider})")
        print(f"{'='*70}\n")

        # Sample queries
        queries = [
            "What is 25 * 4?",
            "Tell me about Python",
            "What's the weather in Tokyo?",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            # For demo, show tool registry
            print("\nAvailable tools:")
            for tool_name in self.tools.tools.keys():
                print(f"  • {tool_name}")

            # Simulate tool call
            if "weather" in query.lower():
                tool_name = "get_weather"
                result = self.tools.call_tool("get_weather", city="Tokyo")
            elif "*" in query or "multiply" in query.lower():
                tool_name = "multiply"
                result = self.tools.call_tool("multiply", a=25, b=4)
            else:
                tool_name = "search_knowledge"
                result = self.tools.call_tool("search_knowledge", query=query)

            print(f"\n[{tool_name}] → {result}")


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
        return "(Ollama not available)"

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
        return f"(Error: {e})"


def call_openai(prompt: str, system_prompt: str = None) -> str:
    """Call OpenAI model."""
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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Main - Demo
# ============================================================================

if __name__ == "__main__":
    print("Function Calling Agent Demo")
    print("=" * 70)

    # Create tool registry
    print("\n🛠️  Creating tool registry...")
    registry = create_tool_registry()
    print(f"✓ Registered {len(registry.tools)} tools\n")

    # Create agent
    agent = FunctionCallingAgent(registry, llm_provider="ollama")

    print("Function Calling Pattern:")
    print("1. LLM receives query and list of available functions/tools")
    print("2. LLM decides which functions to call and with what arguments")
    print("3. Agent executes the function calls")
    print("4. LLM receives results and generates final answer\n")

    print("This pattern powers:")
    print("  ✓ ChatGPT Plugins")
    print("  ✓ Claude Tool Use")
    print("  ✓ OpenAI Function Calling API")
    print("  ✓ Most production LLM systems\n")

    print("Note: Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real execution.\n")

    # Run demo
    agent.run_interactive_demo()

    # Uncomment to process actual queries with LLM:
    # result = agent.process_query("What is 12 times 5?", use_llm=True)
