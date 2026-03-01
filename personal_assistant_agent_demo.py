"""
personal_assistant_agent_demo.py

Demo: Personal Assistant Agent – an LLM-powered agent that manages a
digital assistant workspace including calendar events, to-do lists,
notes, and reminders. This pattern underpins widely-used products like
Google Assistant, Siri, Microsoft Copilot, and many chat-based
productivity tools.

Flow:
1. Maintain simple in-memory structures for events, tasks, and notes
2. Parse natural language commands such as "Schedule meeting", "Add a
   task", or "Remind me" using the LLM
3. Execute operations on the internal state and confirm actions
4. Answer questions by querying the stored data

This form of assistant is extremely popular as a productivity aid and
is advanced when it integrates multiple modalities or services.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python personal_assistant_agent_demo.py

"""

import os
from datetime import datetime
from typing import List, Dict, Any

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
# LLM helpers (consistent with other demos)
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
# Personal assistant infrastructure
# ============================================================================

class PersonalAssistant:
    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.events: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.notes: List[str] = []

    def handle_command(self, command: str) -> str:
        """Use LLM to interpret natural language command and perform actions."""
        prompt = (
            "You are a personal assistant parsing user commands. "
            "Return a JSON object with fields 'action' and parameters. "
            "Actions: add_event, list_events, add_task, list_tasks, "
            "add_note, list_notes, unknown.\n\n"
            f"Command: {command}\n\nReturn JSON:"  
        )
        response = call_llm(prompt, self.llm_provider)
        try:
            obj = json.loads(response)
        except Exception:
            return "Sorry, I didn't understand."

        action = obj.get("action")
        if action == "add_event":
            self.events.append(obj.get("event", {}))
            return "Event added."
        if action == "list_events":
            return json.dumps(self.events, default=str)
        if action == "add_task":
            self.tasks.append(obj.get("task", {}))
            return "Task added."
        if action == "list_tasks":
            return json.dumps(self.tasks)
        if action == "add_note":
            self.notes.append(obj.get("note", ""))
            return "Note added."
        if action == "list_notes":
            return json.dumps(self.notes)
        return "Action not recognized."

    def query(self, question: str) -> str:
        # simple retrieval queries about stored data
        if "what" in question.lower() and "events" in question.lower():
            return json.dumps(self.events, default=str)
        if "tasks" in question.lower():
            return json.dumps(self.tasks)
        if "notes" in question.lower():
            return json.dumps(self.notes)
        return "I can help add/list events, tasks, or notes."


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import json

    assistant = PersonalAssistant(llm_provider="ollama")

    commands = [
        "Schedule a meeting with Bob on March 3 at 2pm.",
        "Add task: buy groceries tomorrow.",
        "Write a note about holiday plans.",
    ]

    for cmd in commands:
        print(f"Command: {cmd}")
        print(assistant.handle_command(cmd))

    print("\nCurrent events:")
    print(assistant.query("what events do I have?"))
    print("\nCurrent tasks:")
    print(assistant.query("show my tasks"))
    print("\nCurrent notes:")
    print(assistant.query("list notes"))
