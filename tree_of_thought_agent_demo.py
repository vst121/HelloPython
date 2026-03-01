"""
tree_of_thought_agent_demo.py

Demo: Tree of Thought (ToT) Agent - Advanced Problem-Solving Pattern.

This agent:
1. GENERATES: Creates multiple possible reasoning paths (thoughts)
2. EVALUATES: Scores each thought for promise and progress
3. EXPLORES: Builds a tree of reasoning paths
4. BACKTRACKS: Abandons low-scoring branches
5. SYNTHESIZES: Finds the best solution path through the tree

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python tree_of_thought_agent_demo.py

This pattern is used in cutting-edge LLM research and systems like Claude,
GPT-4, and other advanced AI systems for complex problem-solving, planning,
code generation, and logical reasoning tasks.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math

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
# Tree of Thought Data Structures
# ============================================================================

class ThoughtStatus(Enum):
    """Status of a thought/node in the tree."""
    UNEXPLORED = "unexplored"
    EXPLORING = "exploring"
    PROMISING = "promising"
    DEAD_END = "dead_end"
    SOLUTION = "solution"


@dataclass
class Thought:
    """A single thought/node in the reasoning tree."""
    id: str
    parent_id: Optional[str]
    content: str
    depth: int
    score: float = 0.0
    status: ThoughtStatus = ThoughtStatus.UNEXPLORED
    children: List['Thought'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def __repr__(self):
        return f"Thought(id={self.id}, depth={self.depth}, score={self.score:.2f}, status={self.status.value})"

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def add_child(self, child: 'Thought'):
        """Add a child thought."""
        self.children.append(child)


class ThoughtTree:
    """Manages the tree of thoughts."""

    def __init__(self, root_content: str):
        self.root = Thought(
            id="root",
            parent_id=None,
            content=root_content,
            depth=0,
        )
        self.all_thoughts: Dict[str, Thought] = {"root": self.root}
        self.thought_counter = 0

    def add_thought(
        self,
        parent_id: str,
        content: str,
        score: float = 0.0,
    ) -> Thought:
        """Add a new thought to the tree."""
        self.thought_counter += 1
        parent = self.all_thoughts[parent_id]
        depth = parent.depth + 1

        thought_id = f"thought_{self.thought_counter}"
        thought = Thought(
            id=thought_id,
            parent_id=parent_id,
            content=content,
            depth=depth,
            score=score,
        )

        parent.add_child(thought)
        self.all_thoughts[thought_id] = thought
        return thought

    def get_best_leaf(self) -> Optional[Thought]:
        """Get the highest-scoring leaf node."""
        leaves = [t for t in self.all_thoughts.values() if t.is_leaf()]
        if not leaves:
            return None
        return max(leaves, key=lambda t: t.score)

    def get_promising_leaves(self, top_k: int = 3) -> List[Thought]:
        """Get top-k promising leaves to explore."""
        leaves = [
            t for t in self.all_thoughts.values()
            if t.is_leaf() and t.status != ThoughtStatus.DEAD_END
        ]
        leaves.sort(key=lambda t: t.score, reverse=True)
        return leaves[:top_k]

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the tree."""
        return {
            "total_nodes": len(self.all_thoughts),
            "max_depth": max(t.depth for t in self.all_thoughts.values()),
            "leaves": len([t for t in self.all_thoughts.values() if t.is_leaf()]),
            "promising": len([
                t for t in self.all_thoughts.values()
                if t.status == ThoughtStatus.PROMISING
            ]),
        }

    def print_tree(self, max_depth: int = 4):
        """Print tree structure."""
        def print_node(node: Thought, prefix: str = ""):
            if node.depth > max_depth:
                return

            status_marker = {
                ThoughtStatus.SOLUTION: "✓",
                ThoughtStatus.PROMISING: "→",
                ThoughtStatus.DEAD_END: "✗",
                ThoughtStatus.EXPLORING: "...",
                ThoughtStatus.UNEXPLORED: "○",
            }[node.status]

            content_preview = node.content[:40].replace("\n", " ")
            print(
                f"{prefix}{status_marker} [{node.id}] ({node.score:.2f}) {content_preview}..."
            )

            for child in node.children:
                print_node(child, prefix + "  ")

        print_node(self.root)


# ============================================================================
# Tree of Thought Agent
# ============================================================================

class TreeOfThoughtAgent:
    """
    Tree of Thought Agent - explores multiple reasoning paths to find
    optimal solutions. Used in cutting-edge LLM systems for complex problems.
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        max_depth: int = 3,
        branching_factor: int = 3,
        beam_width: int = 2,
    ):
        self.llm_provider = llm_provider
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width
        self.tree: Optional[ThoughtTree] = None

    def generate_thoughts(
        self,
        problem: str,
        parent_thought: str,
        num_thoughts: int = 3,
    ) -> List[str]:
        """Generate multiple possible next thoughts/steps."""
        prompt = f"""You are exploring possible solutions to a problem using Tree of Thought reasoning.

PROBLEM: {problem}

CURRENT REASONING: {parent_thought}

Generate {num_thoughts} different possible next thoughts or approaches. 
Consider different angles, strategies, and perspectives.
Format as a numbered list (1., 2., 3., etc.)."""

        system_prompt = """You are a creative problem-solver. Generate diverse, thoughtful alternatives.
Each thought should be a meaningful step toward solving the problem."""

        response = call_llm(prompt, self.llm_provider, system_prompt)

        # Parse thoughts from numbered list
        thoughts = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                # Remove number and period
                thought = line.split(".", 1)[1].strip()
                if thought:
                    thoughts.append(thought)

        return thoughts[:num_thoughts]

    def evaluate_thought(self, problem: str, thought: str) -> float:
        """Evaluate how promising a thought is (0.0 to 1.0)."""
        prompt = f"""Evaluate this reasoning step for solving the problem:

PROBLEM: {problem}

REASONING: {thought}

Score from 0-100 based on:
- Progress toward solution (does it help?)
- Clarity and coherence
- Feasibility
- Originality and insight

Respond with ONLY a number 0-100."""

        system_prompt = "You are an expert evaluator. Be objective and rigorous."

        response = call_llm(prompt, self.llm_provider, system_prompt)

        # Extract score
        try:
            score = float(response.split()[0]) / 100.0
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            return 0.5

    def is_solution(self, problem: str, thought: str) -> bool:
        """Check if a thought is a complete solution."""
        prompt = f"""Does this reasoning completely solve the problem?

PROBLEM: {problem}

REASONING: {thought}

Answer with YES or NO only."""

        system_prompt = "You are a solution validator. Be strict."

        response = call_llm(prompt, self.llm_provider, system_prompt)

        return "yes" in response.lower()

    def explore_tree(self, problem: str) -> Thought:
        """Explore the tree of thoughts to find best solution."""
        print(f"\n{'='*70}")
        print(f"Tree of Thought Agent (using {self.llm_provider})")
        print(f"{'='*70}\n")

        print(f"Problem: {problem}\n")

        # Initialize tree with problem as root
        self.tree = ThoughtTree(problem)
        self.tree.root.status = ThoughtStatus.EXPLORING

        # BFS-style exploration
        depth = 0
        while depth < self.max_depth:
            depth += 1
            print(f"\n{'='*50}")
            print(f"Depth {depth} - Exploring")
            print(f"{'='*50}\n")

            # Get promising leaves to expand
            to_explore = self.tree.get_promising_leaves(self.beam_width)

            if not to_explore:
                # If no promising leaves, explore root
                to_explore = [self.tree.root]

            print(f"Expanding {len(to_explore)} promising branches...\n")

            for parent_thought in to_explore:
                print(f"📍 Parent: {parent_thought.content[:50]}...")

                # Generate child thoughts
                children_contents = self.generate_thoughts(
                    problem,
                    parent_thought.content,
                    num_thoughts=self.branching_factor,
                )

                for child_content in children_contents:
                    # Evaluate
                    score = self.evaluate_thought(problem, child_content)

                    # Add to tree
                    child = self.tree.add_thought(
                        parent_thought.id,
                        child_content,
                        score=score,
                    )

                    # Check if solution
                    if self.is_solution(problem, child_content):
                        child.status = ThoughtStatus.SOLUTION
                        print(f"  ✓ SOLUTION FOUND: {child_content[:50]}...")
                        return child

                    # Mark status
                    if score > 0.7:
                        child.status = ThoughtStatus.PROMISING
                        print(f"  → Promising ({score:.2f}): {child_content[:40]}...")
                    else:
                        child.status = ThoughtStatus.DEAD_END
                        print(f"  ✗ Dead-end ({score:.2f})")

            print(f"\nTree stats: {self.tree.get_tree_stats()}")

        # Return best solution found
        best = self.tree.get_best_leaf()
        if best:
            best.status = ThoughtStatus.SOLUTION
            return best

        return None

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using Tree of Thought."""
        solution = self.explore_tree(problem)

        print(f"\n{'='*70}")
        print("SOLUTION FOUND")
        print(f"{'='*70}\n")

        if solution:
            print(f"✓ Best Solution:\n{solution.content}\n")
            print(f"Confidence Score: {solution.score:.1%}")
            print(f"Reasoning Depth: {solution.depth} steps\n")

            # Trace path to solution
            path = []
            current = solution
            while current:
                path.append(current.content[:40])
                current_id = current.parent_id
                current = self.tree.all_thoughts.get(current_id) if current_id else None

            print("Reasoning Path:")
            for i, step in enumerate(reversed(path)):
                print(f"  {i+1}. {step}...")

        print(f"\nTree Statistics:")
        stats = self.tree.get_tree_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return {
            "problem": problem,
            "solution": solution.content if solution else None,
            "confidence": solution.score if solution else 0.0,
            "depth": solution.depth if solution else 0,
            "tree_size": stats["total_nodes"],
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
    print("Tree of Thought Agent Demo")
    print("=" * 70)

    print("\nTree of Thought (ToT) Algorithm:")
    print("1. GENERATE: Create multiple reasoning paths (thoughts)")
    print("2. EVALUATE: Score each thought for promise")
    print("3. EXPLORE: Build tree of reasoning branches")
    print("4. BACKTRACK: Prune low-scoring branches")
    print("5. SYNTHESIZE: Find best solution path\n")

    print("Key Advantages:")
    print("  ✓ Explores multiple reasoning strategies")
    print("  ✓ Backtracks from dead-end paths")
    print("  ✓ Finds better solutions than single-path reasoning")
    print("  ✓ Useful for complex problems: planning, coding, math\n")

    print("Used by:")
    print("  ✓ OpenAI (GPT-4 advanced reasoning)")
    print("  ✓ Anthropic (Claude complex problem-solving)")
    print("  ✓ Cutting-edge LLM research\n")

    print("Note: Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real execution.\n")

    # Create agent
    agent = TreeOfThoughtAgent(
        llm_provider="ollama",
        max_depth=3,
        branching_factor=3,
        beam_width=2,
    )

    # Example problems
    problems = [
        "How should I structure a Python project for a data science application?",
        "What are the key steps to learning machine learning?",
        "Design a system to solve complex optimization problems",
    ]

    print("Example Problems:")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")

    print("\nTree of Thought Exploration Flow:")
    print("  Depth 1: Generate 3 main approaches")
    print("  ├─ Approach 1: Score 0.85 (promising) → Expand")
    print("  ├─ Approach 2: Score 0.65 (dead-end)")
    print("  └─ Approach 3: Score 0.72 (promising) → Expand")
    print("  ")
    print("  Depth 2: Expand promising branches")
    print("  ├─ From Approach 1:")
    print("  │  ├─ Strategy A: Score 0.88 → ✓ SOLUTION")
    print("  │  ├─ Strategy B: Score 0.72 (promising)")
    print("  │  └─ Strategy C: Score 0.45 (dead-end)")
    print("  └─ From Approach 3: ...\n")

    # Uncomment to run with real LLM:
    # result = agent.solve(problems[0])
    # print(f"\nResult: {json.dumps(result, indent=2)}")
