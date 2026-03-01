"""
multi_agent_demo.py

Demo: Multi-Agent System (Agent Collaboration Pattern).

This system:
1. DELEGATES: Main coordinator breaks down work across specialized agents
2. SPECIALIZATION: Each agent has specific expertise/role
3. COLLABORATION: Agents communicate and share results
4. CONSENSUS: Agents vote or evaluate each other's work
5. SYNTHESIS: Coordinator combines results into final answer

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python multi_agent_demo.py

This pattern powers CrewAI, multi-agent frameworks, and enterprise AI systems.
Features: Specialization, fault-tolerance, distributed problem-solving, quality assurance.
"""

import os
import json
import re
from typing import List, Dict, Optional, Any
from enum import Enum
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
# Agent Roles & Specializations
# ============================================================================

class AgentRole(Enum):
    """Different specialized agent roles."""
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    EXECUTOR = "executor"


@dataclass
class AgentCapability:
    """Describes what an agent can do."""
    role: AgentRole
    expertise: str
    system_prompt: str
    max_tokens: int = 500


# ============================================================================
# Agent Definitions
# ============================================================================

AGENT_CAPABILITIES = {
    AgentRole.RESEARCHER: AgentCapability(
        role=AgentRole.RESEARCHER,
        expertise="Gathers information, researches topics, finds facts",
        system_prompt="""You are a Research Agent. Your job is to:
1. Gather relevant information about the topic
2. Find key facts and data points
3. Identify credible sources and evidence
4. Present findings clearly with sources
Keep responses focused and factual.""",
    ),
    AgentRole.ANALYZER: AgentCapability(
        role=AgentRole.ANALYZER,
        expertise="Analyzes data, identifies patterns, draws insights",
        system_prompt="""You are an Analysis Agent. Your job is to:
1. Analyze provided information deeply
2. Identify patterns, trends, and relationships
3. Draw meaningful insights
4. Consider multiple perspectives
5. Highlight key takeaways
Be thorough but concise.""",
    ),
    AgentRole.CRITIC: AgentCapability(
        role=AgentRole.CRITIC,
        expertise="Evaluates quality, identifies weaknesses, provides constructive feedback",
        system_prompt="""You are a Critic Agent. Your job is to:
1. Evaluate the quality of previous work
2. Identify logical flaws or weaknesses
3. Challenge assumptions
4. Suggest improvements
5. Rate quality on a scale
Be fair but rigorous.""",
    ),
    AgentRole.SYNTHESIZER: AgentCapability(
        role=AgentRole.SYNTHESIZER,
        expertise="Combines multiple perspectives, creates unified answers, resolves conflicts",
        system_prompt="""You are a Synthesis Agent. Your job is to:
1. Review all agent contributions
2. Identify common themes and agreements
3. Resolve any disagreements
4. Create a comprehensive unified answer
5. Ensure all perspectives are represented
Be comprehensive and balanced.""",
    ),
    AgentRole.EXECUTOR: AgentCapability(
        role=AgentRole.EXECUTOR,
        expertise="Implements decisions, takes action, produces deliverables",
        system_prompt="""You are an Executor Agent. Your job is to:
1. Take the final plan/decision
2. Create actionable steps
3. Identify resources needed
4. Provide implementation details
5. Highlight risks and mitigation
Be practical and detailed.""",
    ),
}


# ============================================================================
# Individual Agent
# ============================================================================

class Agent:
    """Specialized AI agent with a specific role."""

    def __init__(self, agent_id: int, role: AgentRole, llm_provider: str = "ollama"):
        self.agent_id = agent_id
        self.role = role
        self.llm_provider = llm_provider
        self.capability = AGENT_CAPABILITIES[role]
        self.memory = []  # Conversation history
        self.output = None

    def __repr__(self):
        return f"Agent({self.agent_id}, {self.role.value})"

    def think(self, prompt: str, context: str = "") -> str:
        """Process prompt with agent's specialization."""
        full_prompt = f"{prompt}"
        if context:
            full_prompt += f"\n\nContext from other agents:\n{context}"

        messages = [
            {"role": "system", "content": self.capability.system_prompt},
            {"role": "user", "content": full_prompt},
        ]

        self.memory.append({"role": "user", "content": prompt})

        response = call_llm_internal(
            messages, self.llm_provider, self.capability.max_tokens
        )

        self.memory.append({"role": "assistant", "content": response})
        self.output = response
        return response

    def get_summary(self) -> Dict[str, Any]:
        """Get agent's summary."""
        return {
            "id": self.agent_id,
            "role": self.role.value,
            "expertise": self.capability.expertise,
            "output": self.output[:200] if self.output else None,
        }


# ============================================================================
# Multi-Agent System (Coordinator)
# ============================================================================

class MultiAgentSystem:
    """Coordinates multiple specialized agents to solve complex problems."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.agents: Dict[AgentRole, Agent] = {}
        self.results: Dict[str, Any] = {}
        self.iteration = 0

    def initialize_agents(self, roles: List[AgentRole]):
        """Create agents for specified roles."""
        for i, role in enumerate(roles, 1):
            self.agents[role] = Agent(i, role, self.llm_provider)

    def broadcast(self, prompt: str) -> Dict[AgentRole, str]:
        """Send prompt to all agents and collect responses."""
        responses = {}
        for role, agent in self.agents.items():
            response = agent.think(prompt)
            responses[role] = response
        return responses

    def collaborative_think(self, prompt: str, iteration: int = 1) -> Dict[AgentRole, str]:
        """Agents think with knowledge of others' outputs."""
        responses = {}
        context_so_far = ""

        for role in [
            AgentRole.RESEARCHER,
            AgentRole.ANALYZER,
            AgentRole.CRITIC,
            AgentRole.EXECUTOR,
        ]:
            if role not in self.agents:
                continue

            agent = self.agents[role]
            response = agent.think(prompt, context_so_far)
            responses[role] = response

            # Build context for next agent
            context_so_far += f"\n[{role.value.upper()}]: {response[:150]}...\n"

        return responses

    def consensus_evaluation(self) -> Dict[str, Any]:
        """Synthesizer reviews all outputs and creates consensus."""
        if AgentRole.SYNTHESIZER not in self.agents:
            return {"status": "No synthesizer agent"}

        synthesis_prompt = "Review all the following contributions and create a unified, comprehensive answer:\n"

        for role, agent in self.agents.items():
            if role != AgentRole.SYNTHESIZER:
                synthesis_prompt += f"\n{role.value.upper()} Agent:\n{agent.output}\n"

        synthesizer = self.agents[AgentRole.SYNTHESIZER]
        final_answer = synthesizer.think(synthesis_prompt)

        return {
            "status": "success",
            "final_answer": final_answer,
            "contributions": {
                role.value: agent.output[:100] for role, agent in self.agents.items()
            },
        }

    def run(self, query: str, use_collaborative: bool = True) -> Dict[str, Any]:
        """Execute multi-agent workflow."""
        print(f"\n{'='*70}")
        print(f"Multi-Agent System (using {self.llm_provider})")
        print(f"{'='*70}")
        print(f"\nQuery: {query}\n")

        # Initialize agents
        roles = [
            AgentRole.RESEARCHER,
            AgentRole.ANALYZER,
            AgentRole.CRITIC,
            AgentRole.SYNTHESIZER,
            AgentRole.EXECUTOR,
        ]

        print("📋 Initializing agents...")
        self.initialize_agents(roles)
        for role in roles:
            print(f"  ✓ {role.value} agent ready")

        # Phase 1: Research & Analysis
        print(f"\n{'='*50}")
        print("Phase 1: Research & Analysis")
        print(f"{'='*50}")

        if use_collaborative:
            print("🔄 Collaborative thinking (sequential context)...")
            responses = self.collaborative_think(query)
        else:
            print("🎯 Broadcasting to all agents...")
            responses = self.broadcast(query)

        for role, response in responses.items():
            print(f"\n[{role.value.upper()}]:")
            print(f"  {response[:150]}...")

        # Phase 2: Evaluation & Synthesis
        print(f"\n{'='*50}")
        print("Phase 2: Consensus & Synthesis")
        print(f"{'='*50}")

        print("🤝 Creating consensus...")
        consensus = self.consensus_evaluation()

        if consensus.get("status") == "success":
            print("\n✓ Consensus reached!")
            print(f"\nFinal Answer:\n{consensus['final_answer']}\n")
        else:
            print(f"Could not reach consensus: {consensus}")

        # Summary
        print(f"\n{'='*70}")
        print("MULTI-AGENT SUMMARY")
        print(f"{'='*70}")
        print(f"Agents deployed: {len(self.agents)}")
        for role, agent in self.agents.items():
            print(f"  • {role.value}: Contributed {len(agent.output) if agent.output else 0} chars")

        return {
            "query": query,
            "agents_used": len(self.agents),
            "final_answer": consensus.get("final_answer"),
            "success": consensus.get("status") == "success",
        }


# ============================================================================
# LLM Integration
# ============================================================================

def call_llm_internal(
    messages: List[Dict], provider: str = "ollama", max_tokens: int = 500
) -> str:
    """Call LLM with message format."""
    if provider.lower() == "openai":
        return call_openai_internal(messages, max_tokens)
    else:
        return call_ollama_internal(messages, max_tokens)


def call_ollama_internal(messages: List[Dict], max_tokens: int = 500) -> str:
    """Call Ollama model."""
    if not OLLAMA_AVAILABLE:
        return "(Ollama not available)"

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3")
    url = f"{host.rstrip('/')}/chat?model={model}"

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


def call_openai_internal(messages: List[Dict], max_tokens: int = 500) -> str:
    """Call OpenAI model."""
    if not OPENAI_AVAILABLE:
        return "(OpenAI not available)"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(No API key)"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Main - Demo
# ============================================================================

if __name__ == "__main__":
    print("Multi-Agent System Demo")
    print("=" * 70)

    # Example query
    query = "What are the best practices for building AI systems in 2026?"

    # Create system
    system = MultiAgentSystem(llm_provider="ollama")

    print("\nNote: This demo shows the multi-agent pattern.")
    print("Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real LLM execution.\n")

    print("Multi-Agent Workflow:")
    print("1. RESEARCHER: Gathers information and facts")
    print("2. ANALYZER: Identifies patterns and insights")
    print("3. CRITIC: Evaluates quality and identifies gaps")
    print("4. EXECUTOR: Creates actionable implementation steps")
    print("5. SYNTHESIZER: Combines all perspectives into unified answer\n")

    # Uncomment to run with real LLM:
    # result = system.run(query, use_collaborative=True)
    # print(f"\nResult: {result}")

    # For now, show expected flow
    print("Expected Output:")
    print("- Each agent contributes from their expertise")
    print("- Agents build on each other's insights (sequential context)")
    print("- Synthesizer creates unified, comprehensive answer")
    print("- Final output incorporates all perspectives\n")
