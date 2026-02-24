"""coroutines_demo.py
Demonstration of generator-based coroutines (PEP 342-style) and cooperative
pipelines using `send()`, `throw()`, and `close()`.

This complements the existing `async_await_demo.py` by showing the older
but still useful generator-coroutine style that enables cooperative
composition (co-creation) of data-processing pipelines.

Key Topics Covered:
- Basic coroutines with yield-based message passing
- Priming coroutines with decorators
- Broadcasting and pipeline patterns
- Exception handling and error propagation
- Bidirectional communication between coroutines
- Stateful coroutines and finite state machines
"""
from functools import wraps
from typing import Callable, Iterable, Any
import time


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
        next(gen)
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
def broadcaster(targets: Iterable):
    """Broadcast values to multiple consumer coroutines."""
    try:
        while True:
            value = (yield)
            for t in targets:
                t.send(value)
    except GeneratorExit:
        for t in targets:
            t.close()
        print("broadcaster: closing and shutting down targets")


def producer(values: Iterable[float], target):
    """Send a sequence of values to a coroutine target."""
    for v in values:
        target.send(v)


@coroutine
def filter_coroutine(predicate: Callable, target):
    """Filter values based on a predicate and forward to target."""
    try:
        while True:
            value = (yield)
            if predicate(value):
                target.send(value)
    except GeneratorExit:
        target.close()
        print("filter: closing")


@coroutine
def transformer(func: Callable, target):
    """Transform values using a function and forward to target."""
    try:
        while True:
            value = (yield)
            transformed = func(value)
            target.send(transformed)
    except GeneratorExit:
        target.close()
        print("transformer: closing")


@coroutine
def accumulator(target, buffer_size: int = 3):
    """Accumulate values and send them as batches to target.
    
    Demonstrates buffering pattern in coroutine pipelines.
    """
    buffer = []
    try:
        while True:
            value = (yield)
            buffer.append(value)
            if len(buffer) >= buffer_size:
                target.send(buffer.copy())
                buffer.clear()
    except GeneratorExit:
        if buffer:
            target.send(buffer)  # Flush remaining items
        target.close()
        print(f"accumulator: closing (flushed {len(buffer)} items)")


@coroutine
def echoing_receiver(name: str = "echo"):
    """Receiver that demonstrates two-way communication via yield value.
    
    Unlike simple receivers, this one returns values received, allowing
    the sender to know what was processed.
    """
    try:
        result = (yield)  # Initial send to prime
        count = 0
        while True:
            count += 1
            result = (yield f"{name} received item #{count}: {result}")
    except GeneratorExit:
        print(f"{name}: received and echoed {count} items")


@coroutine
def error_handler(target, name: str = "error_handler"):
    """Coroutine that catches exceptions and logs them.
    
    Demonstrates exception propagation and error handling in pipelines.
    """
    try:
        while True:
            try:
                value = (yield)
                target.send(value)
            except ValueError as e:
                print(f"{name}: caught ValueError: {e}")
    except GeneratorExit:
        target.close()
        print(f"{name}: closing")


@coroutine
def conditional_router(targets_dict: dict, default_target=None):
    """Route values to different targets based on value type or condition.
    
    Demonstrates conditional pipeline routing.
    """
    try:
        while True:
            value = (yield)
            value_type = type(value).__name__
            target = targets_dict.get(value_type, default_target)
            if target:
                target.send(value)
            else:
                print(f"router: no target for {value_type}")
    except GeneratorExit:
        print("router: closing")
        for target in targets_dict.values():
            if target:
                target.close()
        if default_target:
            default_target.close()


def run_demo():
    print("Generator-based Coroutines Demo (cooperative pipelines)")
    print("=" * 68)

    # DEMO 1: Basic broadcaster pattern
    print("\n[DEMO 1] Basic Broadcaster Pattern")
    print("-" * 68)
    p1 = printer(prefix="Printer1: ")
    p2 = printer(prefix="Printer2: ")
    avg = averager(name="Averager")

    bc = broadcaster([p1, avg])
    sample_values = [10, 20, 30, 25, 15]
    print("Producing values to broadcaster -> printer1 + averager")
    producer(sample_values, bc)

    print("Sending one value directly to Printer2")
    p2.send(99)

    print("Closing broadcaster (this will close p1 and averager)")
    bc.close()
    p2.close()

    # DEMO 2: Filter and transform pipeline
    print("\n[DEMO 2] Filter and Transform Pipeline")
    print("-" * 68)
    
    output = printer(prefix="Filtered & Transformed: ")
    # Chain: transformer -> filter -> printer
    filtered = filter_coroutine(lambda x: x > 20, output)
    transformed = transformer(lambda x: x * 2, filtered)
    
    print("Sending values through transformer -> filter -> printer")
    print("(Only values > 20 will pass through, and they'll be doubled)")
    producer([5, 15, 25, 35, 10, 30], transformed)
    transformed.close()

    # DEMO 3: Accumulator/buffering pattern
    print("\n[DEMO 3] Accumulator/Buffering Pattern")
    print("-" * 68)
    
    def batch_printer(prefix="Batch"):
        try:
            while True:
                batch = (yield)
                print(f"{prefix}: {batch}")
        except GeneratorExit:
            print(f"{prefix}: closed")
    
    batch_target = batch_printer("BatchProcessor")
    buffer = accumulator(batch_target, buffer_size=2)
    
    print("Accumulating values in batches of 2:")
    producer([100, 200, 300, 400, 500], buffer)
    buffer.close()

    # DEMO 4: Two-way communication
    print("\n[DEMO 4] Two-way Communication")
    print("-" * 68)
    
    echo = echoing_receiver("Echo")
    echo.send(None)  # Prime it
    
    print("Sending values and receiving echoes:")
    for i, val in enumerate([42, "hello", 3.14]):
        response = echo.send(val)
        print(f"  -> {response}")
    echo.close()

    # DEMO 5: Error handling in pipelines
    print("\n[DEMO 5] Error Handling in Pipelines")
    print("-" * 68)
    
    safe_printer = printer(prefix="SafePrinter: ")
    error_safe = error_handler(safe_printer, name="ErrorHandler")
    
    print("Sending mixed valid and invalid values:")
    producer([10, 20, 30], error_safe)
    
    # Demonstrate exception injection via throw()
    print("Injecting a ValueError via throw():")
    try:
        error_safe.throw(ValueError, ValueError("Injected error!"))
    except StopIteration:
        pass
    
    error_safe.close()

    # DEMO 6: Conditional routing
    print("\n[DEMO 6] Conditional Routing")
    print("-" * 68)
    
    int_handler = printer(prefix="IntHandler: ")
    str_handler = printer(prefix="StrHandler: ")
    default_handler = printer(prefix="DefaultHandler: ")
    
    router = conditional_router(
        {"int": int_handler, "str": str_handler},
        default_target=default_handler
    )
    
    print("Routing different types to different handlers:")
    mixed_values = [42, "hello", 3.14, 100, "world", True]
    for val in mixed_values:
        router.send(val)
    
    router.close()

    print("\n" + "=" * 68)
    print("Demo complete. Key takeaways:")
    print("✓ Coroutines enable cooperative data pipelines using send()/yield")
    print("✓ @coroutine decorator primes generators for use as consumers")
    print("✓ Pipelines can be chained: producer -> transformer -> filter -> printer")
    print("✓ Two-way communication possible via yield expressions")
    print("✓ Exception handling and routing patterns are supported")
    print("✓ Use .close() and .throw() to manage lifecycle and errors")


if __name__ == "__main__":
    run_demo()
