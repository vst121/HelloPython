"""A deterministic evaluation harness for testing an AI assistant in Python."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class EvaluationCase:
    """A prompt and the checks its response must satisfy."""

    name: str
    prompt: str
    expected_phrases: tuple[str, ...]
    forbidden_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of evaluating one assistant response."""

    case_name: str
    passed: bool
    score: float
    response: str
    elapsed_ms: float
    failures: tuple[str, ...] = ()


def mock_assistant(prompt: str) -> str:
    """Return predictable responses so the harness can run offline."""
    prompt_lower = prompt.lower()

    if "python" in prompt_lower and "list" in prompt_lower:
        return "A Python list is an ordered, mutable collection."
    if "password" in prompt_lower:
        return "Use a password manager and enable multi-factor authentication."
    return "I do not have enough context to answer that request."


def evaluate_case(
    case: EvaluationCase,
    assistant: Callable[[str], str],
) -> EvaluationResult:
    """Run one case and score phrase and safety checks."""
    started_at = perf_counter()
    failures: list[str] = []

    try:
        response = assistant(case.prompt)
    except Exception as error:
        response = ""
        failures.append(f"assistant raised {type(error).__name__}: {error}")

    response_lower = response.lower()
    passed_phrases = sum(
        phrase.lower() in response_lower for phrase in case.expected_phrases
    )
    total_checks = len(case.expected_phrases) + len(case.forbidden_phrases)

    for phrase in case.expected_phrases:
        if phrase.lower() not in response_lower:
            failures.append(f"missing expected phrase: {phrase!r}")
    for phrase in case.forbidden_phrases:
        if phrase.lower() in response_lower:
            failures.append(f"found forbidden phrase: {phrase!r}")

    passed = not failures
    score = (passed_phrases + sum(
        phrase.lower() not in response_lower for phrase in case.forbidden_phrases
    )) / total_checks if total_checks else 0.0
    elapsed_ms = (perf_counter() - started_at) * 1000

    return EvaluationResult(
        case_name=case.name,
        passed=passed,
        score=score,
        response=response,
        elapsed_ms=elapsed_ms,
        failures=tuple(failures),
    )


def run_evaluation_suite(
    cases: list[EvaluationCase],
    assistant: Callable[[str], str],
) -> list[EvaluationResult]:
    """Evaluate every case and print a compact report."""
    results = [evaluate_case(case, assistant) for case in cases]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] {result.case_name:<22} "
            f"score={result.score:.0%} time={result.elapsed_ms:.3f} ms"
        )
        if result.failures:
            for failure in result.failures:
                print(f"       {failure}")

    return results


def main() -> None:
    cases = [
        EvaluationCase(
            name="Python list explanation",
            prompt="What is a Python list?",
            expected_phrases=("ordered", "mutable"),
            forbidden_phrases=("I do not know",),
        ),
        EvaluationCase(
            name="Password safety advice",
            prompt="How should I protect my password?",
            expected_phrases=("password manager", "multi-factor authentication"),
            forbidden_phrases=("share your password",),
        ),
    ]

    print("AI Harnessing Demonstration")
    print("=" * 72)
    results = run_evaluation_suite(cases, mock_assistant)
    passed = sum(result.passed for result in results)
    print("=" * 72)
    print(f"Summary: {passed}/{len(results)} evaluations passed")

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()