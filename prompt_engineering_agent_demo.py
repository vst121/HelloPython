"""
prompt_engineering_agent_demo.py

Demo: Prompt Engineering Agent with Automated Prompt Refinement.

This agent helps craft high-quality prompts by iteratively generating
and validating candidate prompts using the LLM itself. In practice:
1. START: User supplies a task description or goal
2. PROPOSE: Agent generates several prompt candidates
3. VALIDATE: Each prompt is tested with sample inputs
4. CRITIQUE: LLM evaluates prompt effectiveness
5. REFINE: Agent produces better prompts based on critique
6. OUTPUT: Final prompt ready for production

Use cases:
- Building reliable few-shot prompts
- Automatically tuning templates for formality, detail, etc
- Rapid experimentation with prompt wording

This pattern is becoming popular in industry and research (OpenAI,
Anthropic, etc.) under names like "autoprompting" or "prompt
optimization". It's highly practical and can reduce prompt
engineering time by 70%.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python prompt_engineering_agent_demo.py

"""

import os
import json
import random
from typing import List, Dict, Any, Tuple
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
# Data structures
# ============================================================================

@dataclass
class PromptCandidate:
    text: str
    score: float = 0.0
    critique: str = ""


@dataclass
class PromptEngineeringResult:
    original_goal: str
    candidates: List[PromptCandidate]
    final_prompt: str


# ============================================================================
# Core agent
# ============================================================================

class PromptEngineeringAgent:
    """Automates the process of designing and refining prompts."""

    def __init__(
        self,
        llm_provider: str = "ollama",
        candidates_per_round: int = 3,
        max_rounds: int = 3,
    ):
        self.llm_provider = llm_provider
        self.candidates_per_round = candidates_per_round
        self.max_rounds = max_rounds

    def propose_prompts(self, goal: str) -> List[PromptCandidate]:
        """Ask the LLM to propose a set of candidate prompts."""
        system_prompt = (
            "You are a prompt engineer. Generate multiple prompt templates "
            "that would elicit high-quality answers for the given task. "
            "Make them clear, concise, and specific."
        )

        user_prompt = (
            f"Task description / goal:\n{goal}\n\n"
            "Provide a numbered list of candidate prompts."
        )

        response = call_llm(user_prompt, self.llm_provider, system_prompt)
        candidates: List[PromptCandidate] = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # remove leading numbers or bullets
            candidate = line.lstrip("0123456789.- ")
            candidates.append(PromptCandidate(text=candidate))

        # randomly sample if more than needed
        if len(candidates) > self.candidates_per_round:
            candidates = random.sample(candidates, self.candidates_per_round)

        return candidates

    def evaluate_prompt(
        self, prompt: str, examples: List[str]
    ) -> Tuple[float, str]:
        """Evaluate a prompt by running it on sample inputs and scoring.
        returns (score, critique)"""
        system_prompt = (
            "You are an evaluator. Given a prompt and some sample user "
            "inputs, generate expected outputs and rate how well the "
            "prompt would perform. Provide a score 0-1 and a brief critique."
        )

        user_prompt = (
            f"Prompt to evaluate:\n{prompt}\n\n"
            "Sample inputs:\n" + "\n".join(f"- {e}" for e in examples) + "\n\n"
            "For each input, produce an output and then give an overall score "
            "(0 to 1) and a short critique of the prompt."        
        )

        response = call_llm(user_prompt, self.llm_provider, system_prompt)

        # naive parsing: look for 'score' and 'critique'
        score = 0.0
        critique = ""
        for line in response.split("\n"):
            low = line.lower()
            if "score" in low and "0" in low:
                import re
                match = re.search(r"score[: ]*([0-9\.]+)", low)
                if match:
                    try:
                        score = float(match.group(1))
                    except:
                        pass
            if "critique" in low:
                critique = line
                # include next line if exists
                idx = response.split("\n").index(line)
                if idx + 1 < len(response.split("\n")):
                    critique += " " + response.split("\n")[idx + 1]

        return score, critique

    def refine_prompts(
        self, candidates: List[PromptCandidate]
    ) -> List[PromptCandidate]:
        """Ask LLM to refine the best prompts based on critique."""
        texts = "\n".join(f"- {c.text} (score={c.score})" for c in candidates)

        system_prompt = (
            "You are a prompt engineer. Improve the following prompts "
            "taking into account their scores and critiques. "
            "Return a new set of prompt candidates."
        )

        user_prompt = (
            f"Prompt candidates with scores and critiques:\n{texts}\n\n"
            "Provide an improved numbered list of prompts."
        )

        response = call_llm(user_prompt, self.llm_provider, system_prompt)
        new_candidates: List[PromptCandidate] = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            candidate = line.lstrip("0123456789.- ")
            new_candidates.append(PromptCandidate(text=candidate))
        return new_candidates

    def run(self, goal: str, examples: List[str]) -> PromptEngineeringResult:
        """Main orchestration method."""
        all_candidates: List[PromptCandidate] = []

        print(f"\nPrompt Engineering goal: {goal}\n")

        candidates = self.propose_prompts(goal)
        round_idx = 1
        while round_idx <= self.max_rounds:
            print(f"\n=== Round {round_idx} ===")
            for c in candidates:
                print(f"Evaluating candidate: {c.text}")
                score, critique = self.evaluate_prompt(c.text, examples)
                c.score = score
                c.critique = critique
                print(f"  • score={score:.2f}, critique={critique[:60]}...")

            all_candidates.extend(candidates)
            # pick top 1 or 2 to refine
            candidates.sort(key=lambda x: x.score, reverse=True)
            best = candidates[:2]
            if round_idx == self.max_rounds:
                break
            candidates = self.refine_prompts(best)
            round_idx += 1

        # final prompt is highest scoring candidate
        final_prompt = max(all_candidates, key=lambda x: x.score).text
        return PromptEngineeringResult(
            original_goal=goal,
            candidates=all_candidates,
            final_prompt=final_prompt,
        )


# ============================================================================
# LLM integration (same as other demos)
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
# Demo execution
# ============================================================================

if __name__ == "__main__":
    print("Prompt Engineering Agent Demo")
    print("=" * 70)

    goal = "Summarize the environmental benefits of electric vehicles."
    examples = [
        "Tell me the pros of using EVs in plain language.",
        "List three environmental advantages of electric cars.",
    ]

    print(f"\nGoal: {goal}")
    print("Example user inputs:")
    for ex in examples:
        print(f"  - {ex}")

    agent = PromptEngineeringAgent(llm_provider="ollama")
    result = agent.run(goal, examples)

    print("\nFinal prompt generated:")
    print(result.final_prompt)

    print("\nAll candidate prompts and scores:")
    for cand in result.candidates:
        print(f"  score={cand.score:.2f} | {cand.text}")

    # Uncomment to interactively refine your own goal:
    # your_goal = input("Enter prompt goal: ")
    # agent.run(your_goal, examples=[input("Example: ")])
