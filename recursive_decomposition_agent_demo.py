"""
recursive_decomposition_agent_demo.py

Demo: Recursive Decomposition Agent (Hierarchical Problem-Solving Pattern).

This agent:
1. ANALYZES: Examines complexity of the problem
2. DECOMPOSES: Breaks complex problem into simpler sub-problems
3. PRIORITIZES: Orders sub-problems by dependency and difficulty
4. RECURSIVELY SOLVES: Solves each sub-problem (may recurse further)
5. AGGREGATES: Combines sub-solutions into final answer

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python recursive_decomposition_agent_demo.py

This pattern is used in code generation (breaking down programs), planning,
complex reasoning, and any hierarchical problem-solving system.
Used by: GPT-4 code generation, planning systems, complex reasoning tasks.
Features: Divide-and-conquer, recursion, composability, clarity.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
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
# Problem Decomposition Data Structures
# ============================================================================

class ComplexityLevel(Enum):
    """Complexity assessment of a problem."""
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    VERY_COMPLEX = 5


@dataclass
class SubProblem:
    """A sub-problem from decomposition."""
    problem_id: str
    description: str
    complexity: ComplexityLevel
    dependencies: List[str] = field(default_factory=list)
    solution: Optional[str] = None
    is_solved: bool = False
    recursion_depth: int = 0

    def __repr__(self):
        status = "✓" if self.is_solved else "○"
        return f"{status} [{self.problem_id}] {self.description[:40]}... (complexity: {self.complexity.value})"


class DecompositionTree:
    """Manages the tree of decomposed problems."""

    def __init__(self, root_problem: str):
        self.root_id = "root"
        self.problems: Dict[str, SubProblem] = {
            self.root_id: SubProblem(
                problem_id=self.root_id,
                description=root_problem,
                complexity=ComplexityLevel.COMPLEX,
            )
        }
        self.problem_counter = 0
        self.parent_map: Dict[str, str] = {}  # Maps child to parent

    def add_subproblem(
        self,
        parent_id: str,
        description: str,
        complexity: ComplexityLevel,
        dependencies: List[str] = None,
    ) -> str:
        """Add a sub-problem to the tree."""
        self.problem_counter += 1
        problem_id = f"sub_{self.problem_counter}"

        problem = SubProblem(
            problem_id=problem_id,
            description=description,
            complexity=complexity,
            dependencies=dependencies or [],
        )

        self.problems[problem_id] = problem
        self.parent_map[problem_id] = parent_id

        return problem_id

    def set_solution(self, problem_id: str, solution: str):
        """Set solution for a problem."""
        if problem_id in self.problems:
            self.problems[problem_id].solution = solution
            self.problems[problem_id].is_solved = True

    def get_solved_status(self) -> Tuple[int, int]:
        """Get (solved_count, total_count)."""
        solved = sum(1 for p in self.problems.values() if p.is_solved)
        return solved, len(self.problems)

    def get_dependency_order(self) -> List[str]:
        """Get problem IDs in dependency order."""
        ordered = []
        solved = set()

        while len(solved) < len(self.problems):
            for pid, problem in self.problems.items():
                if pid in solved:
                    continue

                # Check if dependencies are satisfied
                if all(dep in solved for dep in problem.dependencies):
                    ordered.append(pid)
                    solved.add(pid)

        return ordered

    def print_tree(self):
        """Print the decomposition tree."""
        def print_problem(pid: str, indent: int = 0):
            p = self.problems[pid]
            prefix = "  " * indent
            status = "✓" if p.is_solved else "○"
            print(f"{prefix}{status} {p.problem_id}: {p.description[:50]}...")

            # Find children
            children = [cid for cid, parent in self.parent_map.items() if parent == pid]
            for child_id in children:
                print_problem(child_id, indent + 1)

        print_problem(self.root_id)


# ============================================================================
# Recursive Decomposition Agent
# ============================================================================

class RecursiveDecompositionAgent:
    """
    Recursive Decomposition Agent - Breaks complex problems into simpler
    sub-problems and solves them hierarchically. Used in code generation,
    planning, and complex reasoning tasks.
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        max_recursion_depth: int = 3,
        complexity_threshold: int = 2,
    ):
        self.llm_provider = llm_provider
        self.max_recursion_depth = max_recursion_depth
        self.complexity_threshold = complexity_threshold
        self.tree: Optional[DecompositionTree] = None
        self.solution_cache: Dict[str, str] = {}

    def assess_complexity(self, problem: str) -> ComplexityLevel:
        """Assess problem complexity using LLM."""
        prompt = f"""Assess the complexity of this problem on a scale of 1-5:
1 = Trivial (one step, obvious answer)
2 = Simple (straightforward, a few steps)
3 = Moderate (requires planning, multiple steps)
4 = Complex (multiple components, interactions)
5 = Very Complex (many dependencies, non-obvious solution)

Problem: {problem}

Respond with ONLY a number 1-5."""

        system_prompt = "You are a complexity assessor. Be objective and concise."

        response = call_llm(prompt, self.llm_provider, system_prompt)

        try:
            level = int(response.strip())
            return ComplexityLevel(max(1, min(5, level)))
        except (ValueError, AttributeError):
            return ComplexityLevel.MODERATE

    def decompose_problem(
        self,
        problem: str,
        max_subproblems: int = 4,
    ) -> List[Tuple[str, ComplexityLevel]]:
        """Decompose a problem into sub-problems."""
        prompt = f"""Break down this problem into {max_subproblems} simpler sub-problems.
Each sub-problem should be:
- Solvable independently or with minimal dependencies
- Clearer than the original problem
- A clear step toward the solution

Problem: {problem}

Format as numbered list:
1. [Sub-problem 1]
2. [Sub-problem 2]
etc.

Only output the list, no other text."""

        system_prompt = """You are an expert at breaking down complex problems.
Create clear, well-scoped sub-problems that can be solved independently."""

        response = call_llm(prompt, self.llm_provider, system_prompt)

        subproblems = []
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                # Remove number and period
                subproblem = line.split(".", 1)[1].strip()
                if subproblem:
                    # Assess complexity
                    complexity = self.assess_complexity(subproblem)
                    subproblems.append((subproblem, complexity))

        return subproblems[:max_subproblems]

    def solve_atomic_problem(self, problem: str) -> str:
        """Solve an atomic (non-decomposable) problem."""
        prompt = f"""Solve this problem clearly and concisely:

Problem: {problem}

Provide:
1. Your approach
2. Step-by-step solution
3. Final answer"""

        system_prompt = "You are an expert problem solver. Be clear and thorough."

        solution = call_llm(prompt, self.llm_provider, system_prompt)
        return solution

    def combine_solutions(
        self,
        original_problem: str,
        subproblem_solutions: List[Tuple[str, str]],
    ) -> str:
        """Combine sub-problem solutions into final answer."""
        solutions_text = ""
        for subproblem, solution in subproblem_solutions:
            solutions_text += f"\nSub-problem: {subproblem}\nSolution: {solution[:100]}...\n"

        prompt = f"""Combine these sub-problem solutions into a comprehensive answer:

Original Problem: {original_problem}

Sub-problem Solutions:{solutions_text}

Create a unified, cohesive final solution that:
1. Addresses the original problem fully
2. Integrates the sub-solutions logically
3. Is clear and well-structured
4. Highlights how sub-solutions connect"""

        system_prompt = """You are an expert at synthesis. Create clear, integrated solutions 
from component parts."""

        final_solution = call_llm(prompt, self.llm_provider, system_prompt)
        return final_solution

    def solve_recursive(
        self,
        problem: str,
        problem_id: str = "root",
        depth: int = 0,
        parent_id: Optional[str] = None,
    ) -> str:
        """Recursively solve a problem through decomposition."""
        # Check cache
        if problem_id in self.solution_cache:
            return self.solution_cache[problem_id]

        # Base case: max recursion depth or simple problem
        complexity = self.assess_complexity(problem)

        if depth >= self.max_recursion_depth or complexity.value <= self.complexity_threshold:
            print(f"{'  '*depth}[Depth {depth}] Solving atomic: {problem[:40]}...")
            solution = self.solve_atomic_problem(problem)
            self.solution_cache[problem_id] = solution

            if self.tree and parent_id:
                self.tree.set_solution(problem_id, solution)

            return solution

        # Recursive case: decompose and solve sub-problems
        print(f"{'  '*depth}[Depth {depth}] Decomposing: {problem[:50]}... (complexity: {complexity.value})")

        subproblems = self.decompose_problem(problem)
        print(f"{'  '*depth}Found {len(subproblems)} sub-problems\n")

        subproblem_solutions = []

        for i, (subproblem, sub_complexity) in enumerate(subproblems, 1):
            sub_id = f"{problem_id}_sub{i}"

            if self.tree:
                self.tree.add_subproblem(
                    problem_id,
                    subproblem,
                    sub_complexity,
                )

            print(f"{'  '*(depth+1)}Solving sub-problem {i}: {subproblem[:40]}...")

            # Recursive call
            subsolution = self.solve_recursive(
                subproblem,
                sub_id,
                depth + 1,
                parent_id=problem_id,
            )

            subproblem_solutions.append((subproblem, subsolution))
            print(f"{'  '*(depth+1)}✓ Sub-problem {i} solved\n")

        # Combine solutions
        print(f"{'  '*depth}Combining {len(subproblem_solutions)} sub-solutions...")
        final_solution = self.combine_solutions(problem, subproblem_solutions)

        self.solution_cache[problem_id] = final_solution

        if self.tree:
            self.tree.set_solution(problem_id, final_solution)

        return final_solution

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using recursive decomposition."""
        print(f"\n{'='*70}")
        print(f"Recursive Decomposition Agent (using {self.llm_provider})")
        print(f"{'='*70}\n")

        print(f"Problem: {problem}\n")

        # Initialize tree
        self.tree = DecompositionTree(problem)
        self.solution_cache.clear()

        # Assess initial complexity
        complexity = self.assess_complexity(problem)
        print(f"Complexity Assessment: {complexity.value}/5\n")

        # Solve recursively
        final_solution = self.solve_recursive(problem)

        # Print results
        print(f"\n{'='*70}")
        print("SOLUTION")
        print(f"{'='*70}\n")

        print(final_solution)

        # Tree stats
        solved, total = self.tree.get_solved_status()
        print(f"\n\nDecomposition Tree Stats:")
        print(f"  Total sub-problems: {total}")
        print(f"  Solved: {solved}")
        print(f"  Cache size: {len(self.solution_cache)}\n")

        return {
            "problem": problem,
            "solution": final_solution,
            "subproblems_count": total,
            "recursion_depth": self.max_recursion_depth,
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
            model="gpt-3.5-turbo", messages=messages, max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Main - Demo
# ============================================================================

if __name__ == "__main__":
    print("Recursive Decomposition Agent Demo")
    print("=" * 70)

    print("\nRecursive Decomposition Algorithm:")
    print("1. ASSESS: Evaluate problem complexity")
    print("2. DECOMPOSE: Break into simpler sub-problems")
    print("3. PRIORITIZE: Order by dependencies")
    print("4. RECURSE: Apply to sub-problems (if still complex)")
    print("5. SOLVE: Solve atomic problems directly")
    print("6. AGGREGATE: Combine sub-solutions\n")

    print("Key Advantages:")
    print("  ✓ Handles arbitrarily complex problems")
    print("  ✓ Clear, structured reasoning process")
    print("  ✓ Sub-solutions can be cached/reused")
    print("  ✓ Easy to understand decomposition tree")
    print("  ✓ Works for code generation, planning, reasoning\n")

    print("Used by:")
    print("  ✓ GPT-4 code generation")
    print("  ✓ Planning and task decomposition systems")
    print("  ✓ Complex reasoning applications")
    print("  ✓ Software architecture design\n")

    print("Note: Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real execution.\n")

    # Create agent
    agent = RecursiveDecompositionAgent(
        llm_provider="ollama",
        max_recursion_depth=3,
        complexity_threshold=2,
    )

    # Example problems
    problems = [
        "Write a Python function that sorts a list of integers using merge sort",
        "Design a recommendation system for an e-commerce platform",
        "Create a plan to learn machine learning from scratch",
    ]

    print("Example Problems:")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")

    print("\n\nExample Decomposition Tree:")
    print("""
Problem: Write a Python function for merge sort
├─ Complexity: 4/5 (COMPLEX) → Decompose
├─ Sub-problem 1: Understand merge sort algorithm (complexity: 2)
│  └─ SOLVE: Merge sort divides array in half, sorts, then merges
├─ Sub-problem 2: Implement array splitting logic (complexity: 2)
│  └─ SOLVE: Use slicing [low:mid] and [mid:high]
├─ Sub-problem 3: Implement merge operation (complexity: 3) → Decompose
│  ├─ Compare two sorted arrays (complexity: 2) → SOLVE
│  └─ Combine results in order (complexity: 2) → SOLVE
└─ Sub-problem 4: Implement recursive function (complexity: 2)
   └─ SOLVE: Base case + recursive calls

Final Solution: Complete merge sort implementation
    """)

    # Uncomment to run with real LLM:
    # result = agent.solve(problems[0])
    # print(f"\nResult: {json.dumps(result, indent=2)}")
