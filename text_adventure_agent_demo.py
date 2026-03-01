"""
text_adventure_agent_demo.py

Demo: Text Adventure Agent – an LLM-driven agent that plays a simple
text-based game by generating actions and interpreting environment
descriptions. Text adventures (e.g. Zork, AI Dungeon) are a classic and
popular use case for generative agents, demonstrating planning, world
modeling, and long-term memory.

Flow:
1. Environment represents rooms, objects, and state
2. Agent receives observation text from environment
3. LLM produces an action (e.g., "go north", "take key")
4. Simulator updates state, returns new description
5. Repeat until goal reached or max steps

This pattern is widely used in research on LLM decision-making, open
ended play, and embodied agents (TextWorld, Jericho).

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python text_adventure_agent_demo.py

"""

import os
import random
from typing import Dict, Tuple

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = False
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Simple text world environment
# ---------------------------------------------------------------------------

class Room:
    def __init__(self, description: str, items: list = None, exits: dict = None):
        self.description = description
        self.items = items or []
        self.exits = exits or {}


class TextWorld:
    def __init__(self):
        # define few connected rooms
        self.rooms: Dict[str, Room] = {
            "hall": Room(
                "You are in a grand hall. There is a door to the north.",
                items=["key"],
                exits={"north": "study"},
            ),
            "study": Room(
                "You are in a cozy study. There's a locked chest here.",
                items=[],
                exits={"south": "hall", "east": "garden"},
            ),
            "garden": Room(
                "You are in a sunny garden. A fountain lies in the center.",
                items=["flower"],
                exits={"west": "study"},
            ),
        }
        self.current_room = "hall"
        self.inventory = []
        self.chest_locked = True

    def observe(self) -> str:
        room = self.rooms[self.current_room]
        obs = room.description
        if room.items:
            obs += " You see " + ", ".join(room.items) + "."
        if self.inventory:
            obs += " You have " + ", ".join(self.inventory) + "."
        return obs

    def step(self, action: str) -> Tuple[str, bool]:
        # returns (observation, done)
        action = action.lower().strip()
        room = self.rooms[self.current_room]

        # movement
        if action.startswith("go "):
            direction = action.split(" ")[1]
            if direction in room.exits:
                self.current_room = room.exits[direction]
                return self.observe(), False
            else:
                return "You can't go that way.", False

        # take item
        if action.startswith("take "):
            item = action.split(" ")[1]
            if item in room.items:
                room.items.remove(item)
                self.inventory.append(item)
                return f"You pick up the {item}.", False
            else:
                return f"There is no {item} here.", False

        # unlock chest
        if action == "open chest":
            if self.current_room == "study":
                if "key" in self.inventory:
                    if self.chest_locked:
                        self.chest_locked = False
                        return "You open the chest and find treasure! You win!", True
                    else:
                        return "The chest is already open.", False
                else:
                    return "The chest is locked. You need a key.", False
            else:
                return "There is no chest here.", False

        return "I don't understand that action.", False


# ---------------------------------------------------------------------------
# Agent using LLM to choose actions
# ---------------------------------------------------------------------------

class TextAdventureAgent:
    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider

    def choose_action(self, observation: str) -> str:
        prompt = (
            "You are playing a text adventure game. Based on the current "
            "observation, choose one simple action (e.g., 'go north', 'take key', "
            "'open chest'). Provide only the action text.\n\n"
            f"Observation: {observation}\n\nAction:"
        )
        action = call_llm(prompt, self.llm_provider)
        # take first line as action
        return action.split("\n")[0]


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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=500)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Demo execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = TextWorld()
    agent = TextAdventureAgent(llm_provider="ollama")

    done = False
    steps = 0
    print("Starting text adventure. Type 'quit' to exit early.\n")
    while not done and steps < 20:
        obs = env.observe()
        print(f"Observation: {obs}")
        action = agent.choose_action(obs)
        print(f"Agent action: {action}")
        if action.lower() == "quit":
            break
        obs, done = env.step(action)
        print(f"Result: {obs}\n")
        steps += 1

    if done:
        print("Game ended: success!")
    else:
        print("Game ended: max steps reached or quit.")
