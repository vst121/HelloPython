"""A small, dependency-free harness for exercising Python functions."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable


@dataclass
class HarnessCase:
    """A named function call and the result it is expected to produce."""

    name: str
    function: Callable[..., Any]
    args: tuple[Any, ...]
    expected: Any


def run_case(case: HarnessCase) -> bool:
    """Run one case, report its duration, and return whether it passed."""
    started_at = perf_counter()

    try:
        actual = case.function(*case.args)
        passed = actual == case.expected
        detail = f"expected {case.expected!r}, got {actual!r}"
    except Exception as error:  # A harness should report failures, not stop early.
        passed = False
        detail = f"raised {type(error).__name__}: {error}"

    elapsed_ms = (perf_counter() - started_at) * 1000
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {case.name:<24} {elapsed_ms:7.3f} ms | {detail}")
    return passed


def add_numbers(first: int, second: int) -> int:
    """Example subject under test."""
    return first + second


def normalize_name(name: str) -> str:
    """Example subject under test with a small input transformation."""
    return " ".join(name.strip().title().split())


def main() -> None:
    cases = [
        HarnessCase("add positive numbers", add_numbers, (2, 3), 5),
        HarnessCase("add negative numbers", add_numbers, (-4, 1), -3),
        HarnessCase("normalize whitespace", normalize_name, ("  ada   lovelace ",), "Ada Lovelace"),
    ]

    print("Python Harnessing Demonstration")
    print("=" * 72)
    passed = sum(run_case(case) for case in cases)
    print("=" * 72)
    print(f"Summary: {passed}/{len(cases)} cases passed")

    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()