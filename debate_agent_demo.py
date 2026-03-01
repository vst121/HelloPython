"""
debate_agent_demo.py

Demo: Debate Agent (Adversarial Reasoning Pattern).

This system:
1. PROPOSES: Initial position/argument from one agent
2. CHALLENGES: Other agents critique and propose alternatives
3. DEFENDS: Original agent responds to criticism
4. DELIBERATES: Agents present evidence and reasoning
5. CONVERGES: Iterative debate until consensus or high confidence

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python debate_agent_demo.py

This pattern is used by Anthropic (Constitutional AI), OpenAI (debate systems),
and AI safety research. It improves answer quality through adversarial testing
and forces agents to defend their reasoning with evidence.
Features: Diverse perspectives, error detection, reasoning verification, consensus-building.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

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
# Debate Agent Roles & Perspectives
# ============================================================================

class AgentPerspective(Enum):
    """Different perspectives agents can take."""
    ADVOCATE = "advocate"  # Supports the initial position
    SKEPTIC = "skeptic"  # Questions and challenges
    ENGINEER = "engineer"  # Practical/implementation perspective
    SCIENTIST = "scientist"  # Evidence-based approach
    ETHICIST = "ethicist"  # Values and implications focus


@dataclass
class DebatePosition:
    """A position taken by an agent in the debate."""
    agent_id: str
    perspective: AgentPerspective
    position: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
    round: int


class DebateAgent:
    """A single agent in a debate with a specific perspective."""

    def __init__(
        self,
        agent_id: str,
        perspective: AgentPerspective,
        llm_provider: str = "ollama",
    ):
        self.agent_id = agent_id
        self.perspective = perspective
        self.llm_provider = llm_provider
        self.positions: List[DebatePosition] = []

    def get_system_prompt(self) -> str:
        """Get system prompt based on perspective."""
        prompts = {
            AgentPerspective.ADVOCATE: """You are an Advocate Agent. Your role is to:
1. Support and strengthen the proposed position
2. Find evidence and reasoning that supports it
3. Address counterarguments constructively
4. Build a convincing case with evidence
Be thorough and thoughtful, but fair.""",
            AgentPerspective.SKEPTIC: """You are a Skeptic Agent. Your role is to:
1. Question assumptions in the position
2. Identify potential flaws and weaknesses
3. Propose alternative perspectives
4. Ask probing questions
Be rigorous but constructive.""",
            AgentPerspective.ENGINEER: """You are an Engineer Agent. Your role is to:
1. Focus on practical implementation
2. Identify feasibility challenges
3. Suggest concrete improvements
4. Consider resource requirements
Be practical and solution-oriented.""",
            AgentPerspective.SCIENTIST: """You are a Scientist Agent. Your role is to:
1. Demand evidence-based reasoning
2. Identify logical gaps
3. Propose empirical approaches
4. Reference data and studies
Be rigorous and evidence-focused.""",
            AgentPerspective.ETHICIST: """You are an Ethicist Agent. Your role is to:
1. Consider ethical implications
2. Identify value conflicts
3. Propose ethical frameworks
4. Ensure fairness and responsibility
Be thoughtful about impact.""",
        }
        return prompts.get(self.perspective, "")

    def take_position(
        self,
        topic: str,
        context: str = "",
        round_num: int = 1,
    ) -> DebatePosition:
        """Generate a position on the topic."""
        if context:
            prompt = f"""Topic: {topic}

Previous debate context:
{context}

From your {self.perspective.value} perspective, what is your position?
Include:
1. Your main argument
2. Key evidence or reasoning
3. Your confidence level (0-100)

Structure your response with clear sections."""
        else:
            prompt = f"""Topic: {topic}

From your {self.perspective.value} perspective, what is your position?
Include:
1. Your main argument
2. Key evidence or reasoning
3. Your confidence level (0-100)

Structure your response with clear sections."""

        response = call_llm(prompt, self.llm_provider, self.get_system_prompt())

        # Parse response (simple extraction)
        confidence = self.extract_confidence(response)
        evidence = self.extract_evidence(response)

        position = DebatePosition(
            agent_id=self.agent_id,
            perspective=self.perspective,
            position=response,
            confidence=confidence,
            evidence=evidence,
            round=round_num,
        )

        self.positions.append(position)
        return position

    def extract_confidence(self, response: str) -> float:
        """Extract confidence score from response."""
        try:
            lines = response.lower().split("\n")
            for line in lines:
                if "confidence" in line and "%" in line:
                    # Extract number
                    import re
                    match = re.search(r"(\d+)", line)
                    if match:
                        return float(match.group(1)) / 100.0
        except Exception:
            pass
        return 0.5

    def extract_evidence(self, response: str) -> List[str]:
        """Extract evidence points from response."""
        evidence = []
        lines = response.split("\n")
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                evidence.append(line)

        return evidence[:3]  # Top 3 evidence points


# ============================================================================
# Debate Moderator
# ============================================================================

class DebateModerator:
    """Manages the debate between agents."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.agents: Dict[AgentPerspective, DebateAgent] = {}
        self.debate_transcript: List[Dict[str, Any]] = []
        self.round_count = 0

    def add_agent(self, perspective: AgentPerspective) -> DebateAgent:
        """Add an agent with a specific perspective."""
        agent = DebateAgent(
            agent_id=f"agent_{perspective.value}",
            perspective=perspective,
            llm_provider=self.llm_provider,
        )
        self.agents[perspective] = agent
        return agent

    def get_debate_context(self, round_num: int) -> str:
        """Get context from previous rounds."""
        if round_num == 1:
            return ""

        context = f"Previous arguments (Round {round_num - 1}):\n"
        for record in self.debate_transcript[-len(self.agents):]:
            agent_id = record["agent_id"]
            position = record["position"][:100]
            context += f"\n{agent_id}: {position}...\n"

        return context

    def conduct_round(self, topic: str, round_num: int = 1) -> List[DebatePosition]:
        """Conduct one round of debate."""
        print(f"\n{'='*60}")
        print(f"Debate Round {round_num}")
        print(f"{'='*60}\n")

        positions = []
        context = self.get_debate_context(round_num)

        for perspective in [
            AgentPerspective.ADVOCATE,
            AgentPerspective.SKEPTIC,
            AgentPerspective.ENGINEER,
            AgentPerspective.SCIENTIST,
            AgentPerspective.ETHICIST,
        ]:
            agent = self.agents[perspective]
            print(f"🎤 {perspective.value.upper()} Agent...")

            position = agent.take_position(topic, context, round_num)

            print(f"Position: {position.position[:80]}...")
            print(f"Confidence: {position.confidence:.0%}\n")

            # Record in transcript
            self.debate_transcript.append({
                "round": round_num,
                "agent_id": agent.agent_id,
                "perspective": perspective.value,
                "position": position.position,
                "confidence": position.confidence,
            })

            positions.append(position)

        return positions

    def evaluate_consensus(self) -> Dict[str, Any]:
        """Evaluate whether consensus has been reached."""
        if not self.debate_transcript:
            return {"consensus": False, "agreement_level": 0.0}

        # Get latest round
        latest_positions = [
            p for p in self.debate_transcript
            if p["round"] == max(p["round"] for p in self.debate_transcript)
        ]

        avg_confidence = sum(p["confidence"] for p in latest_positions) / len(
            latest_positions
        )

        # Simple heuristic: consensus if avg confidence > 0.75
        return {
            "consensus": avg_confidence > 0.75,
            "agreement_level": avg_confidence,
            "confidence_spread": max(
                p["confidence"] for p in latest_positions
            ) - min(p["confidence"] for p in latest_positions),
        }

    def synthesize_conclusion(self, topic: str) -> str:
        """Synthesize a final conclusion from the debate."""
        # Get all arguments
        arguments_text = "Key arguments from the debate:\n"
        for perspective in [
            AgentPerspective.ADVOCATE,
            AgentPerspective.SKEPTIC,
            AgentPerspective.ENGINEER,
            AgentPerspective.SCIENTIST,
            AgentPerspective.ETHICIST,
        ]:
            agent = self.agents[perspective]
            if agent.positions:
                latest = agent.positions[-1]
                arguments_text += f"\n{perspective.value}: {latest.position[:150]}...\n"

        prompt = f"""Based on this debate about:
{topic}

{arguments_text}

Synthesize a fair, balanced conclusion that:
1. Acknowledges valid points from each perspective
2. Identifies areas of agreement
3. Highlights remaining disagreements
4. Provides a practical path forward
5. Acknowledges limitations and uncertainties"""

        system_prompt = """You are a skilled debate moderator. Create balanced, fair conclusions
that respectfully represent all perspectives while identifying the strongest reasoning."""

        conclusion = call_llm(prompt, self.llm_provider, system_prompt)
        return conclusion

    def run_debate(
        self,
        topic: str,
        max_rounds: int = 2,
    ) -> Dict[str, Any]:
        """Run a full debate."""
        print(f"\n{'='*70}")
        print(f"Debate Agent System (using {self.llm_provider})")
        print(f"{'='*70}")
        print(f"\nTopic: {topic}\n")

        # Initialize agents
        print("Initializing debate participants...")
        perspectives = [
            AgentPerspective.ADVOCATE,
            AgentPerspective.SKEPTIC,
            AgentPerspective.ENGINEER,
            AgentPerspective.SCIENTIST,
            AgentPerspective.ETHICIST,
        ]

        for perspective in perspectives:
            self.add_agent(perspective)
            print(f"  ✓ {perspective.value} agent ready")

        # Conduct rounds
        for round_num in range(1, max_rounds + 1):
            self.round_count = round_num
            self.conduct_round(topic, round_num)

            # Check for consensus
            consensus_eval = self.evaluate_consensus()
            print(f"\nConsensus Status: {consensus_eval['agreement_level']:.0%}")

            if consensus_eval["consensus"]:
                print("Consensus reached! Concluding debate.\n")
                break

        # Synthesize conclusion
        print(f"\n{'='*70}")
        print("DEBATE CONCLUSION")
        print(f"{'='*70}\n")

        conclusion = self.synthesize_conclusion(topic)
        print(conclusion)

        return {
            "topic": topic,
            "rounds": self.round_count,
            "participants": len(self.agents),
            "conclusion": conclusion,
            "consensus": consensus_eval.get("consensus", False),
            "agreement_level": consensus_eval.get("agreement_level", 0),
            "transcript_length": len(self.debate_transcript),
        }


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
    print("Debate Agent System Demo")
    print("=" * 70)

    print("\nDebate Format:")
    print("5 Agents with different perspectives argue a topic:")
    print("  1. ADVOCATE - Supports the position")
    print("  2. SKEPTIC - Questions and challenges")
    print("  3. ENGINEER - Practical implementation focus")
    print("  4. SCIENTIST - Evidence-based reasoning")
    print("  5. ETHICIST - Values and implications\n")

    print("Debate Process:")
    print("1. Each agent presents their perspective")
    print("2. Build on previous arguments (context from prior rounds)")
    print("3. Evaluate consensus level across agents")
    print("4. Synthesize fair, balanced conclusion\n")

    print("Benefits:")
    print("  ✓ Surfaces multiple valid viewpoints")
    print("  ✓ Challenges assumptions through disagreement")
    print("  ✓ Improves answer quality through adversarial testing")
    print("  ✓ Identifies logical flaws and gaps")
    print("  ✓ Reaches well-reasoned consensus\n")

    print("Used by:")
    print("  ✓ Anthropic (Constitutional AI, AI safety)")
    print("  ✓ OpenAI (debate systems research)")
    print("  ✓ AI alignment and safety research\n")

    print("Note: Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real execution.\n")

    # Create moderator
    moderator = DebateModerator(llm_provider="ollama")

    # Example topics
    topics = [
        "Should AI be open-source or closed-source?",
        "What is the best approach to AI safety?",
        "How should society regulate artificial intelligence?",
    ]

    print("Example Debate Topics:")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")

    print("\nDebate Flow Example:")
    print("  Round 1: Opening positions from all 5 perspectives")
    print("  ├─ Advocate: 'Position should be adopted because...'")
    print("  ├─ Skeptic: 'But there are concerns...'")
    print("  ├─ Engineer: 'Implementation challenges include...'")
    print("  ├─ Scientist: 'Evidence suggests...'")
    print("  └─ Ethicist: 'Ethical implications are...'\n")
    print("  Round 2: Agents build on each other's arguments")
    print("  └─ Increased consensus (77% agreement)")
    print("\n  Conclusion: Fair synthesis acknowledging all viewpoints\n")

    # Uncomment to run with real LLM:
    # result = moderator.run_debate(topics[0], max_rounds=2)
    # print(f"\nFinal Result: {json.dumps(result, indent=2)}")
