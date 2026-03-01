"""
self_refining_agent_demo.py

Demo: Self-Refining Agent (Iterative Improvement Pattern).

This agent:
1. GENERATES: Creates initial output/response
2. CRITIQUES: Evaluates its own output for quality, accuracy, clarity
3. IDENTIFIES: Finds specific weaknesses and areas for improvement
4. REFINES: Revises the output based on self-critique
5. VALIDATES: Checks if improved output meets quality threshold
6. REPEATS: Continues until output is satisfactory or max iterations reached

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python self_refining_agent_demo.py

This pattern powers modern LLM quality improvements and is used by OpenAI,
Anthropic, and leading AI companies for producing high-quality outputs.
Features: Iterative refinement, self-critique, quality assurance, progressive improvement.
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any
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
# Quality Metrics & Evaluation
# ============================================================================

class QualityMetric(Enum):
    """Quality dimensions to evaluate."""
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    STRUCTURE = "structure"
    RELEVANCE = "relevance"


@dataclass
class QualityScore:
    """Score for a single quality dimension."""
    metric: QualityMetric
    score: float  # 0.0 to 1.0
    feedback: str
    needs_improvement: bool

    def __repr__(self):
        stars = "⭐" * int(self.score * 5)
        return f"{self.metric.value}: {stars} ({self.score:.1%}) - {self.feedback}"


@dataclass
class CritiqueResult:
    """Result of self-critique evaluation."""
    overall_score: float
    quality_scores: List[QualityScore]
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    ready_for_delivery: bool

    def __repr__(self):
        return f"Critique(score={self.overall_score:.1%}, ready={self.ready_for_delivery})"


# ============================================================================
# Self-Refining Agent
# ============================================================================

class SelfRefiningAgent:
    """
    Self-Refining Agent that iteratively improves outputs through
    self-critique and refinement. Powers high-quality AI systems.
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        max_refinement_iterations: int = 3,
        quality_threshold: float = 0.80,
    ):
        self.llm_provider = llm_provider
        self.max_refinement_iterations = max_refinement_iterations
        self.quality_threshold = quality_threshold
        self.iteration_history: List[Dict[str, Any]] = []
        self.current_iteration = 0

    def generate_initial_response(self, query: str) -> str:
        """Generate initial response to the query."""
        prompt = f"""Answer the following question clearly and comprehensively:

Question: {query}

Provide a well-structured, detailed response."""

        system_prompt = "You are a knowledgeable assistant. Provide clear, accurate responses."

        response = call_llm(prompt, self.llm_provider, system_prompt)
        return response

    def self_critique(self, query: str, response: str) -> CritiqueResult:
        """Critique the response for quality dimensions."""
        critique_prompt = f"""Evaluate the following response to this question:

QUESTION: {query}

RESPONSE: {response}

Provide a detailed critique covering:
1. ACCURACY (0-100): How accurate and factually correct is this?
2. CLARITY (0-100): How clear and easy to understand?
3. COMPLETENESS (0-100): Does it address all aspects of the question?
4. STRUCTURE (0-100): How well organized and structured?
5. RELEVANCE (0-100): How relevant is the content?

Format your response as JSON with this structure:
{{
  "overall_score": <average of scores>,
  "accuracy": {{"score": X, "feedback": "..."}},
  "clarity": {{"score": X, "feedback": "..."}},
  "completeness": {{"score": X, "feedback": "..."}},
  "structure": {{"score": X, "feedback": "..."}},
  "relevance": {{"score": X, "feedback": "..."}},
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "improvement_suggestions": ["...", "...", "..."],
  "ready_for_delivery": <boolean>
}}"""

        system_prompt = """You are a rigorous quality evaluator. Provide honest, constructive critique.
Output ONLY valid JSON, no other text."""

        critique_json = call_llm(critique_prompt, self.llm_provider, system_prompt)

        # Parse critique
        try:
            data = json.loads(critique_json)
            
            quality_scores = []
            for metric_name in ["accuracy", "clarity", "completeness", "structure", "relevance"]:
                if metric_name in data:
                    m_data = data[metric_name]
                    score = m_data.get("score", 0) / 100.0
                    feedback = m_data.get("feedback", "")
                    quality_scores.append(
                        QualityScore(
                            metric=QualityMetric[metric_name.upper()],
                            score=score,
                            feedback=feedback,
                            needs_improvement=score < 0.8,
                        )
                    )

            return CritiqueResult(
                overall_score=data.get("overall_score", 0) / 100.0,
                quality_scores=quality_scores,
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                improvement_suggestions=data.get("improvement_suggestions", []),
                ready_for_delivery=data.get("ready_for_delivery", False),
            )
        except Exception as e:
            # Fallback critique
            return CritiqueResult(
                overall_score=0.5,
                quality_scores=[],
                strengths=["Generated response"],
                weaknesses=[f"Critique parsing error: {e}"],
                improvement_suggestions=["Review manually"],
                ready_for_delivery=False,
            )

    def refine_response(
        self,
        query: str,
        current_response: str,
        critique: CritiqueResult,
        iteration: int,
    ) -> str:
        """Refine response based on critique."""
        weaknesses_text = "\n".join([f"- {w}" for w in critique.weaknesses])
        suggestions_text = "\n".join([f"- {s}" for s in critique.improvement_suggestions])

        refine_prompt = f"""Refine and improve the following response based on the critique:

ORIGINAL QUESTION: {query}

CURRENT RESPONSE: {current_response}

IDENTIFIED WEAKNESSES:
{weaknesses_text}

IMPROVEMENT SUGGESTIONS:
{suggestions_text}

OVERALL SCORE: {critique.overall_score:.1%}

Create an improved version that addresses the weaknesses and suggestions.
Focus on making it more accurate, clear, complete, well-structured, and relevant."""

        system_prompt = """You are an expert writer and editor. Refine responses to be 
exceptionally clear, accurate, and comprehensive. Output only the refined response."""

        refined = call_llm(refine_prompt, self.llm_provider, system_prompt)
        return refined

    def process_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Full self-refining pipeline."""
        print(f"\n{'='*70}")
        print(f"Self-Refining Agent (using {self.llm_provider})")
        print(f"{'='*70}")
        print(f"\nQuery: {query}\n")

        current_response = None
        best_critique = None
        best_response = None

        for iteration in range(1, self.max_refinement_iterations + 1):
            self.current_iteration = iteration
            print(f"{'='*50}")
            print(f"Iteration {iteration}/{self.max_refinement_iterations}")
            print(f"{'='*50}\n")

            # Step 1: Generate or refine
            if iteration == 1:
                print("📝 Generating initial response...")
                current_response = self.generate_initial_response(query)
            else:
                print("✏️  Refining response based on critique...")
                current_response = self.refine_response(
                    query, current_response, best_critique, iteration
                )

            if verbose:
                print(f"Response: {current_response[:150]}...\n")

            # Step 2: Critique
            print("🔍 Self-critiquing response...")
            critique = self.self_critique(query, current_response)

            if verbose:
                print(f"Overall Quality Score: {critique.overall_score:.1%}")
                print(f"Ready for delivery: {critique.ready_for_delivery}\n")

                if critique.quality_scores:
                    print("Quality Breakdown:")
                    for qs in critique.quality_scores:
                        print(f"  {qs}")

                if critique.weaknesses:
                    print(f"\nWeaknesses: {', '.join(critique.weaknesses[:2])}")

            best_critique = critique
            best_response = current_response

            # Check if ready
            if critique.overall_score >= self.quality_threshold:
                print(f"\n✓ Quality threshold reached! ({critique.overall_score:.1%} >= {self.quality_threshold:.1%})")
                break

            if iteration < self.max_refinement_iterations:
                print(f"Continuing to next iteration for further improvement...\n")

        # Summary
        print(f"\n{'='*70}")
        print("REFINEMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Total iterations: {self.current_iteration}")
        print(f"Final quality score: {best_critique.overall_score:.1%}")
        print(f"Target threshold: {self.quality_threshold:.1%}")
        print(f"Status: {'✓ READY' if best_critique.overall_score >= self.quality_threshold else '⚠ NEEDS WORK'}\n")

        print(f"Final Response:\n{best_response}\n")

        return {
            "query": query,
            "final_response": best_response,
            "iterations": self.current_iteration,
            "final_quality_score": best_critique.overall_score,
            "strengths": best_critique.strengths,
            "weaknesses": best_critique.weaknesses,
            "ready_for_delivery": best_critique.overall_score >= self.quality_threshold,
        }

    def compare_iterations(self) -> str:
        """Show comparison of iterations."""
        if not self.iteration_history:
            return "No iteration history recorded."

        report = "\nIteration Comparison:\n"
        for i, record in enumerate(self.iteration_history, 1):
            report += f"\nIteration {i}:\n"
            report += f"  Score: {record.get('score', 0):.1%}\n"
            report += f"  Response: {record.get('response', '')[:50]}...\n"

        return report


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
    print("Self-Refining Agent Demo")
    print("=" * 70)

    print("\nSelf-Refinement Pipeline:")
    print("1. GENERATE: Create initial response")
    print("2. CRITIQUE: Evaluate across quality dimensions")
    print("   - Accuracy, Clarity, Completeness, Structure, Relevance")
    print("3. IDENTIFY: Find specific weaknesses and improvement areas")
    print("4. REFINE: Revise based on feedback")
    print("5. VALIDATE: Check if quality threshold is met")
    print("6. REPEAT: Continue until satisfied or max iterations\n")

    print("This pattern is used by:")
    print("  ✓ OpenAI (response refinement)")
    print("  ✓ Anthropic (quality improvements)")
    print("  ✓ Production LLM systems (output quality assurance)\n")

    print("Note: Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real execution.\n")

    # Create agent
    agent = SelfRefiningAgent(
        llm_provider="ollama",
        max_refinement_iterations=3,
        quality_threshold=0.80,
    )

    # Example query
    example_query = "Explain how machine learning works in simple terms"

    print("Example Query:")
    print(f"  \"{example_query}\"\n")

    print("Expected Workflow:")
    print("  Iteration 1: Generate initial response → Critique → Score: ~60%")
    print("  Iteration 2: Refine → Critique → Score: ~75%")
    print("  Iteration 3: Refine again → Critique → Score: ~85%")
    print("  ✓ Quality threshold (80%) achieved!\n")

    # Uncomment to run with real LLM:
    # result = agent.process_query(example_query, verbose=True)
    # print(f"Result: {json.dumps(result, indent=2)}")
