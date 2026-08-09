"""A deterministic evaluation harness for testing an AI assistant in Python."""

import json
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


@dataclass(frozen=True)
class RubricCriterion:
    """A weighted quality check for a response."""

    name: str
    check: Callable[[str], bool]
    weight: float = 1.0


@dataclass(frozen=True)
class RubricResult:
    """The weighted outcome of evaluating a response against a rubric."""

    score: float
    passed: bool
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdvancedEvaluationCase:
    """A case with optional schema, rubric, latency, and consistency checks."""

    name: str
    prompt: str
    rubric: tuple[RubricCriterion, ...] = ()
    required_json_keys: tuple[str, ...] = ()
    max_latency_ms: float | None = None
    repetitions: int = 1


@dataclass(frozen=True)
class AdvancedEvaluationResult:
    """The outcome of an advanced evaluation case."""

    case_name: str
    passed: bool
    score: float
    responses: tuple[str, ...]
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


def structured_mock_assistant(prompt: str) -> str:
    """Return a predictable JSON response for contract testing."""
    if "summarize" in prompt.lower():
        return json.dumps(
            {
                "summary": "Python lists preserve order and can be changed.",
                "confidence": 0.98,
            }
        )
    return json.dumps({"summary": "No summary available.", "confidence": 0.2})


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


def evaluate_rubric(
    response: str,
    criteria: tuple[RubricCriterion, ...],
) -> RubricResult:
    """Score a response against weighted, user-defined quality criteria."""
    if not criteria:
        return RubricResult(score=1.0, passed=True)

    failures: list[str] = []
    total_weight = sum(criterion.weight for criterion in criteria)
    earned_weight = 0.0

    for criterion in criteria:
        if criterion.weight <= 0:
            raise ValueError(f"criterion weight must be positive: {criterion.name}")
        try:
            passed = criterion.check(response)
        except Exception as error:
            passed = False
            failures.append(
                f"criterion {criterion.name!r} raised {type(error).__name__}: {error}"
            )
        if passed:
            earned_weight += criterion.weight
        else:
            failures.append(f"rubric criterion failed: {criterion.name}")

    score = earned_weight / total_weight
    return RubricResult(score=score, passed=not failures, failures=tuple(failures))


def validate_json_contract(
    response: str,
    required_keys: tuple[str, ...],
) -> tuple[bool, str]:
    """Validate that a response is a JSON object containing required keys."""
    if not required_keys:
        return True, ""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        return False, f"invalid JSON: {error.msg}"
    if not isinstance(payload, dict):
        return False, "JSON response must be an object"

    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        return False, f"missing JSON keys: {', '.join(missing_keys)}"
    return True, ""


def evaluate_advanced_case(
    case: AdvancedEvaluationCase,
    assistant: Callable[[str], str],
) -> AdvancedEvaluationResult:
    """Run rubric, JSON contract, latency, and consistency checks."""
    if case.repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    started_at = perf_counter()
    responses: list[str] = []
    failures: list[str] = []

    for _ in range(case.repetitions):
        try:
            responses.append(assistant(case.prompt))
        except Exception as error:
            failures.append(f"assistant raised {type(error).__name__}: {error}")

    elapsed_ms = (perf_counter() - started_at) * 1000
    if not responses:
        return AdvancedEvaluationResult(
            case_name=case.name,
            passed=False,
            score=0.0,
            responses=(),
            elapsed_ms=elapsed_ms,
            failures=tuple(failures),
        )

    rubric_results = [evaluate_rubric(response, case.rubric) for response in responses]
    score = sum(result.score for result in rubric_results) / len(rubric_results)
    for result in rubric_results:
        failures.extend(result.failures)

    contract_passed, contract_failure = validate_json_contract(
        responses[0], case.required_json_keys
    )
    if not contract_passed:
        failures.append(contract_failure)

    if case.max_latency_ms is not None and elapsed_ms > case.max_latency_ms:
        failures.append(
            f"latency budget exceeded: {elapsed_ms:.3f} ms > "
            f"{case.max_latency_ms:.3f} ms"
        )

    if len(set(responses)) > 1:
        failures.append("inconsistent responses across repetitions")

    return AdvancedEvaluationResult(
        case_name=case.name,
        passed=not failures,
        score=score,
        responses=tuple(responses),
        elapsed_ms=elapsed_ms,
        failures=tuple(failures),
    )


def run_advanced_suite(
    cases: list[AdvancedEvaluationCase],
    assistant: Callable[[str], str],
) -> list[AdvancedEvaluationResult]:
    """Run advanced cases and print their diagnostics."""
    results = [evaluate_advanced_case(case, assistant) for case in cases]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] {result.case_name:<22} "
            f"score={result.score:.0%} time={result.elapsed_ms:.3f} ms"
        )
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

    advanced_cases = [
        AdvancedEvaluationCase(
            name="Structured summary contract",
            prompt="Summarize Python lists as JSON.",
            rubric=(
                RubricCriterion(
                    "explains ordering",
                    lambda response: "preserve order" in response,
                    weight=2.0,
                ),
                RubricCriterion(
                    "mentions mutability",
                    lambda response: "changed" in response,
                ),
            ),
            required_json_keys=("summary", "confidence"),
            max_latency_ms=100,
            repetitions=2,
        )
    ]

    print("\nAdvanced Harness Demonstration")
    print("=" * 72)
    advanced_results = run_advanced_suite(advanced_cases, structured_mock_assistant)
    advanced_passed = sum(result.passed for result in advanced_results)
    print("=" * 72)
    print(
        f"Advanced summary: {advanced_passed}/{len(advanced_results)} "
        "evaluations passed"
    )

    if passed != len(results) or advanced_passed != len(advanced_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()