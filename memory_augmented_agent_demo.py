"""
memory_augmented_agent_demo.py

Demo: Memory-Augmented Agent (Stateful Conversation Pattern).

This agent:
1. CAPTURES: Stores important facts, context, and user preferences
2. RETRIEVES: Accesses relevant memory during conversation
3. INTEGRATES: Blends memory with current query for context
4. LEARNS: Updates memory with new information
5. PERSONALIZES: Adapts responses based on accumulated knowledge

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python memory_augmented_agent_demo.py

This pattern powers personal assistants, chatbots, customer support AI,
and any system that needs to maintain context over multi-turn conversations.
Used by: ChatGPT custom instructions, Claude projects, personal AI assistants.
Features: Long-term memory, context awareness, personalization, relationship building.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

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
# Memory Management
# ============================================================================

@dataclass
class MemoryEntry:
    """A single entry in the agent's memory."""
    entry_id: str
    category: str  # fact, preference, goal, constraint, history
    content: str
    importance: float  # 0.0 to 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    relevance_count: int = 0  # How many times used


class MemoryBank:
    """Manages the agent's long-term memory."""

    def __init__(self, max_entries: int = 100):
        self.entries: Dict[str, MemoryEntry] = {}
        self.max_entries = max_entries
        self.entry_counter = 0

    def add_memory(
        self,
        category: str,
        content: str,
        importance: float = 0.5,
    ) -> str:
        """Add a new memory entry."""
        self.entry_counter += 1
        entry_id = f"mem_{self.entry_counter}"

        entry = MemoryEntry(
            entry_id=entry_id,
            category=category,
            content=content,
            importance=importance,
        )

        self.entries[entry_id] = entry

        # Remove least important if over capacity
        if len(self.entries) > self.max_entries:
            self._prune_least_important()

        return entry_id

    def _prune_least_important(self):
        """Remove least important memory when at capacity."""
        # Score: importance * relevance_count
        entries_with_scores = [
            (eid, (e.importance * max(1, e.relevance_count)))
            for eid, e in self.entries.items()
        ]
        entries_with_scores.sort(key=lambda x: x[1])

        # Remove bottom 10%
        to_remove = max(1, len(self.entries) // 10)
        for eid, _ in entries_with_scores[:to_remove]:
            del self.entries[eid]

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories for a query."""
        query_lower = query.lower()
        scored_entries = []

        for entry in self.entries.values():
            # Simple relevance scoring: keyword matching
            score = 0.0

            # Check category relevance
            if entry.category in query_lower:
                score += 0.3

            # Check content relevance (keyword overlap)
            query_words = set(query_lower.split())
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            score += overlap * 0.5

            # Boost by importance and recency of use
            score += entry.importance * 0.2

            if score > 0:
                scored_entries.append((entry, score))

        # Sort by score
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        # Mark as accessed
        for entry, _ in scored_entries[:top_k]:
            entry.last_accessed = datetime.now().isoformat()
            entry.relevance_count += 1

        return [entry for entry, _ in scored_entries[:top_k]]

    def get_memory_summary(self) -> str:
        """Get a summary of the memory bank for the agent."""
        if not self.entries:
            return "No memories stored yet."

        summary = f"Memory Bank ({len(self.entries)} entries):\n"

        # Group by category
        by_category = {}
        for entry in self.entries.values():
            if entry.category not in by_category:
                by_category[entry.category] = []
            by_category[entry.category].append(entry)

        for category, entries in by_category.items():
            summary += f"\n{category.upper()}:\n"
            for entry in entries[:3]:  # Show top 3 per category
                summary += f"  • {entry.content[:60]}... (importance: {entry.importance:.1f})\n"

        return summary

    def clear_memory(self):
        """Clear all memory."""
        self.entries.clear()
        self.entry_counter = 0


# ============================================================================
# Conversation History
# ============================================================================

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_id: int
    user_message: str
    agent_response: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    memories_used: List[str] = field(default_factory=list)
    memories_created: List[str] = field(default_factory=list)


class ConversationHistory:
    """Manages conversation history."""

    def __init__(self, max_turns: int = 50):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns

    def add_turn(
        self,
        user_message: str,
        agent_response: str,
        memories_used: List[str] = None,
        memories_created: List[str] = None,
    ) -> ConversationTurn:
        """Add a conversation turn."""
        turn = ConversationTurn(
            turn_id=len(self.turns) + 1,
            user_message=user_message,
            agent_response=agent_response,
            memories_used=memories_used or [],
            memories_created=memories_created or [],
        )

        self.turns.append(turn)

        # Trim if over capacity
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        return turn

    def get_context_window(self, num_turns: int = 5) -> str:
        """Get recent conversation context."""
        relevant_turns = self.turns[-num_turns:]

        context = "Recent conversation:\n"
        for turn in relevant_turns:
            context += f"User: {turn.user_message}\n"
            context += f"Agent: {turn.agent_response[:100]}...\n\n"

        return context


# ============================================================================
# Memory-Augmented Agent
# ============================================================================

class MemoryAugmentedAgent:
    """
    Memory-Augmented Agent that maintains context across conversations.
    Powers personal assistants, chatbots, and stateful AI systems.
    """

    def __init__(
        self,
        agent_name: str = "Assistant",
        llm_provider: str = "ollama",
    ):
        self.agent_name = agent_name
        self.llm_provider = llm_provider
        self.memory_bank = MemoryBank(max_entries=100)
        self.conversation_history = ConversationHistory(max_turns=50)
        self.user_id = None

    def extract_memories(self, message: str, response: str) -> Tuple[List[str], List[str]]:
        """Extract important facts and preferences from conversation."""
        prompt = f"""Analyze this conversation turn and identify important information to remember:

User: {message}
Agent: {response}

Identify:
1. FACTS: Important information about the user or topic
2. PREFERENCES: User preferences and styles
3. GOALS: User's objectives or plans

Format as:
FACT: ...
PREFERENCE: ...
GOAL: ...

Only list items worth remembering (skip trivial details)."""

        system_prompt = """You are a memory extraction specialist. Identify important, 
non-obvious information worth storing for future conversations."""

        extraction = call_llm(prompt, self.llm_provider, system_prompt)

        # Parse and store
        memories_created = []

        lines = extraction.split("\n")
        for line in lines:
            if line.startswith("FACT:"):
                content = line.replace("FACT:", "").strip()
                if content:
                    mem_id = self.memory_bank.add_memory("fact", content, importance=0.8)
                    memories_created.append(mem_id)

            elif line.startswith("PREFERENCE:"):
                content = line.replace("PREFERENCE:", "").strip()
                if content:
                    mem_id = self.memory_bank.add_memory(
                        "preference", content, importance=0.7
                    )
                    memories_created.append(mem_id)

            elif line.startswith("GOAL:"):
                content = line.replace("GOAL:", "").strip()
                if content:
                    mem_id = self.memory_bank.add_memory("goal", content, importance=0.9)
                    memories_created.append(mem_id)

        return memories_created

    def respond(self, user_message: str) -> str:
        """Respond to user with memory augmentation."""
        # Retrieve relevant memories
        relevant_memories = self.memory_bank.retrieve_relevant(user_message, top_k=5)
        memory_ids = [m.entry_id for m in relevant_memories]

        # Build context
        memory_context = ""
        if relevant_memories:
            memory_context = "Relevant knowledge about the user/context:\n"
            for mem in relevant_memories:
                memory_context += f"- ({mem.category}) {mem.content}\n"

        # Build prompt with context
        recent_context = self.conversation_history.get_context_window(num_turns=3)

        prompt = f"""{memory_context}

{recent_context}

User: {user_message}

Respond naturally while incorporating the stored knowledge above. 
Be personalized, contextual, and build on previous conversations."""

        system_prompt = f"""You are {self.agent_name}, a personalized AI assistant. 
You have memory of previous conversations and user preferences.
Use that knowledge to provide helpful, personalized responses.
If you learn new important information, acknowledge it naturally."""

        response = call_llm(prompt, self.llm_provider, system_prompt)

        # Extract and store new memories
        memories_created = self.extract_memories(user_message, response)

        # Record conversation turn
        self.conversation_history.add_turn(
            user_message,
            response,
            memories_used=memory_ids,
            memories_created=memories_created,
        )

        return response

    def multi_turn_conversation(self, messages: List[str]) -> List[str]:
        """Run a multi-turn conversation."""
        print(f"\n{'='*70}")
        print(f"Memory-Augmented Agent: {self.agent_name}")
        print(f"{'='*70}\n")

        responses = []

        for i, message in enumerate(messages, 1):
            print(f"{'='*50}")
            print(f"Turn {i}")
            print(f"{'='*50}\n")

            print(f"User: {message}")

            response = self.respond(message)
            responses.append(response)

            print(f"\nAgent: {response}\n")

            # Show memory state
            recent_memories = self.memory_bank.retrieve_relevant(message, top_k=2)
            if recent_memories:
                print("Memories used:")
                for mem in recent_memories:
                    print(f"  • ({mem.category}) {mem.content[:50]}...")

            if i < len(messages):
                print("\n")

        return responses

    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return {
            "agent_name": self.agent_name,
            "total_turns": len(self.conversation_history.turns),
            "memory_entries": len(self.memory_bank.entries),
            "memory_summary": self.memory_bank.get_memory_summary(),
        }

    def print_state(self):
        """Print agent state."""
        state = self.get_agent_state()
        print(f"\n{'='*70}")
        print("AGENT STATE")
        print(f"{'='*70}\n")
        print(f"Agent: {state['agent_name']}")
        print(f"Conversation turns: {state['total_turns']}")
        print(f"Memory entries: {state['memory_entries']}\n")
        print(state['memory_summary'])


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
            model="gpt-3.5-turbo", messages=messages, max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Main - Demo
# ============================================================================

if __name__ == "__main__":
    print("Memory-Augmented Agent Demo")
    print("=" * 70)

    print("\nMemory-Augmented Architecture:")
    print("1. CAPTURE: Extract facts, preferences, goals from conversation")
    print("2. RETRIEVE: Find relevant memories for context")
    print("3. INTEGRATE: Blend memories with current query")
    print("4. LEARN: Update memory with new information")
    print("5. PERSONALIZE: Adapt responses based on history\n")

    print("Key Benefits:")
    print("  ✓ Maintains context over multi-turn conversations")
    print("  ✓ Personalized responses based on user history")
    print("  ✓ Remembers preferences, goals, and constraints")
    print("  ✓ Relationship building over time")
    print("  ✓ More coherent long-term conversations\n")

    print("Used by:")
    print("  ✓ ChatGPT (custom instructions, conversation memory)")
    print("  ✓ Claude (projects feature with persistent context)")
    print("  ✓ Personal AI assistants")
    print("  ✓ Customer support chatbots\n")

    print("Note: Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real execution.\n")

    # Create agent
    agent = MemoryAugmentedAgent(
        agent_name="Maya",
        llm_provider="ollama",
    )

    # Sample multi-turn conversation
    conversation = [
        "Hi! I'm Alex, a software engineer interested in AI. I prefer concise responses.",
        "I'm currently learning Python and want to build an ML project about computer vision.",
        "Can you suggest resources for learning computer vision with Python?",
        "I work best with hands-on tutorials and have about 5 hours per week to dedicate.",
    ]

    print("Example Multi-Turn Conversation:")
    for i, msg in enumerate(conversation, 1):
        print(f"\n{i}. User: {msg}")

    print("\n\nExpected Agent Behavior:")
    print("Turn 1: Greets Alex, notes engineer + AI interest")
    print("        Memory: [PREFERENCE: concise responses]")
    print("\nTurn 2: Learns about Python + ML + vision goals")
    print("        Memory: [GOAL: build computer vision project]")
    print("\nTurn 3: Retrieves goals + preferences, gives concise advice")
    print("        Memory: [INTEREST: computer vision with Python]")
    print("\nTurn 4: Uses all previous context for personalized recommendations")
    print("        Memory: [AVAILABILITY: 5 hours/week, STYLE: hands-on]\n")

    # Uncomment to run with real LLM:
    # agent.multi_turn_conversation(conversation)
    # agent.print_state()
