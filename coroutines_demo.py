"""coroutines_demo.py
Demonstration of generator-based coroutines (PEP 342-style) and cooperative
pipelines using `send()`, `throw()`, and `close()`.

This complements the existing `async_await_demo.py` by showing the older
but still useful generator-coroutine style that enables cooperative
composition (co-creation) of data-processing pipelines.
"""
from functools import wraps
from typing import Callable, Iterable, Any, List


def coroutine(func: Callable) -> Callable:
    """Decorator to prime generator-based coroutines automatically.

    Usage:
        @coroutine
        def receiver():
            ...

    The returned object is primed (advanced to first `yield`) so it's
    ready to receive values with `.send()`.
    """

    @wraps(func)
    def primed(*args, **kwargs):
        gen = func(*args, **kwargs)
        # Prime the generator so it's ready to receive with .send()
        try:
            next(gen)
        except StopIteration:
            # In case the generator exits immediately, just return it
            return gen
        return gen

    return primed


@coroutine
def printer(prefix: str = ""):
    """Simple coroutine that prints received values with an optional prefix."""
    try:
        while True:
            value = (yield)
            print(f"{prefix}{value}")
    except GeneratorExit:
        print(f"{prefix}printer: closing")


@coroutine
def averager(name: str = "avg"):
    """Coroutine that keeps a running average of numbers sent to it.

    It demonstrates stateful coroutines that co-create computation when
    connected into a pipeline.
    """
    total = 0.0
    count = 0
    try:
        while True:
            value = (yield)
            total += value
            count += 1
            print(f"{name}: received={value:.2f} running_avg={(total / count):.2f}")
    except GeneratorExit:
        print(f"{name}: closing (processed {count} items)")


@coroutine
def broadcaster(targets: Iterable[Any]):
    """Broadcast values to multiple consumer coroutines.

    Robustness improvements:
    - Accepts any iterable of targets and keeps an internal list.
    - If a target raises on send (closed or runtime error), the error is
      reported and the target is removed so the pipeline keeps running.
    - On close, attempts to close remaining targets gracefully.
    """
    targets_list: List[Any] = list(targets)
    try:
        while True:
            value = (yield)
            alive: List[Any] = []
            for t in targets_list:
                try:
                    t.send(value)
                    alive.append(t)
                except (StopIteration, GeneratorExit, RuntimeError):
                    # Target closed or cannot accept values anymore: drop it
                    continue
                except Exception as exc:
                    # Log and keep target (transient errors shouldn't drop it)
                    print(f"broadcaster: target {getattr(t,'__name__',t.__class__.__name__)} error: {exc}")
                    alive.append(t)
            targets_list = alive
    except GeneratorExit:
        for t in targets_list:
            try:
                t.close()
            except Exception:
                pass
        print("broadcaster: closing and shutting down targets")


def producer(values: Iterable[float], target: Any) -> int:
    """Send a sequence of values to a coroutine target.

    Returns the number of values successfully sent.
    """
    sent = 0
    for v in values:
        try:
            target.send(v)
            sent += 1
        except Exception as exc:
            print(f"producer: failed to send {v!r} -> {exc}")
            break
    return sent


def run_demo():
    print("Generator-based Coroutines Demo (cooperative pipelines)")
    print("=" * 68)

    # Consumers
    p1 = printer(prefix="Printer1: ")
    p2 = printer(prefix="Printer2: ")
    avg = averager(name="Averager")

    # Broadcast incoming values to multiple consumers
    bc = broadcaster([p1, avg])

    # Produce some values into the broadcaster
    sample_values = [10, 20, 30, 25, 15]
    print("Producing values to broadcaster -> printer1 + averager")
    sent = producer(sample_values, bc)
    print(f"producer: sent {sent} items")

    # Now directly send to a second printer
    print("Sending one value directly to Printer2")
    p2.send(99)

    # Demonstrate graceful shutdown (closing the broadcaster will close targets)
    print("Closing broadcaster (this will close p1 and averager)")
    bc.close()

    # Printer2 is still open; close it explicitly
    p2.close()

    print("Demo complete. Key notes:")
    print("- Coroutines enable cooperative data pipelines using send()/yield.")
    print("- Decorator `@coroutine` primes generators for use as consumers.")
    print("- Use `.close()` and `.throw()` to manage coroutine lifecycle and errors.")


if __name__ == "__main__":
    run_demo()
