"""
persona_agent_demo.py

Demo: Persona-based Chat Agent – the LLM adopts a character or persona
for the conversation. Changing persona is a common feature in popular
chatbots and roleplaying systems (e.g. ChatGPT with character modes,
AI Dungeon, Replika).

Flow:
1. Initialize with a set of predefined personas (friendly, expert,
l   humorous, formal, etc.)
2. User selects or switches persona mid-chat
3. Agent uses system prompt templates to maintain persona consistency
4. Conversation history preserved for context

This pattern is widely used for entertainment, education, and
therapeutic chatbots.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python persona_agent_demo.py

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
# Persona definitions
# ---------------------------------------------------------------------------

class Persona:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


PERSONAS = {
    "friendly": Persona(
        "friendly",
        "You are a warm and friendly assistant who uses casual language, "
        "offers encouragement, and occasionally uses emojis."
    ),
    "expert": Persona(
        "expert",
        "You are a knowledgeable expert who provides detailed, precise "
        "answers filled with facts and technical depth."
    ),
    "humorous": Persona(
        "humorous",
        "You are a witty chatbot that answers with jokes and light humor "
        "while still being informative."
    ),
    "formal": Persona(
        "formal",
        "You are a polite and formal assistant, using proper grammar and "
        "a respectful tone."
    ),
}


# ---------------------------------------------------------------------------
# Chat session with persona
# ---------------------------------------------------------------------------

class PersonaChatSession:
    def __init__(self, persona: Persona, llm_provider: str = "ollama"):
        self.persona = persona
        self.llm_provider = llm_provider
        self.history: List[dict] = []  # store messages as {role,content}

    def system_prompt(self) -> str:
        return f"Persona: {self.persona.description}\n" \
               "Maintain this persona throughout the conversation. " \
               "If the user asks you to switch personas, comply."

    def send_message(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        messages = [
            {"role": "system", "content": self.system_prompt()}
        ]
        messages.extend(self.history)

        response = call_llm_with_history(messages, self.llm_provider)
        self.history.append({"role": "assistant", "content": response})
        return response

    def switch_persona(self, new_persona_name: str) -> bool:
        if new_persona_name in PERSONAS:
            self.persona = PERSONAS[new_persona_name]
            # option: clear history or keep for context
            self.history = []
            return True
        return False


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm_with_history(messages: List[dict], provider: str = "ollama") -> str:
    if provider.lower() == "openai":
        return call_openai_history(messages)
    else:
        return call_ollama_history(messages)


def call_ollama_history(messages: List[dict]) -> str:
    if not OLLAMA_AVAILABLE:
        return "(Ollama unavailable)"
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3")
    url = f"{host.rstrip('/')}/chat?model={model}"
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


def call_openai_history(messages: List[dict]) -> str:
    if not OPENAI_AVAILABLE:
        return "(OpenAI unavailable)"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(No API key)"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1500)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Persona-based Chat Agent Demo")
    print("=" * 60)

    session = PersonaChatSession(PERSONAS["friendly"], llm_provider="ollama")

    print("Current persona: friendly")
    print(session.send_message("Hello, who are you?"))

    print("\nSwitching to expert persona...")
    session.switch_persona("expert")
    print(session.send_message("Now explain quantum entanglement."))

    print("\nSwitching to humorous persona...")
    session.switch_persona("humorous")
    print(session.send_message("Tell me a joke about computers."))

    print("\nConversation history in current session:")
    for msg in session.history:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
