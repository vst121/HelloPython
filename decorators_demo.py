"""decorators_demo.py
Comprehensive demonstration of Python decorators and their patterns.

Decorators are higher-order functions that modify or enhance functions/classes
without permanently changing their source code. They enable:
- Cross-cutting concerns (logging, timing, caching)
- Function/class enhancement
- Behavior modification
- Composition and chaining

Key Topics Covered:
- Function decorators with and without arguments
- Class decorators
- Decorator chaining and stacking
- Preserving function metadata with functools.wraps
- Practical patterns: timing, memoization, validation, logging
"""

import time
import functools
from typing import Callable, Any, TypeVar, Tuple
from collections import defaultdict

F = TypeVar('F', bound=Callable[..., Any])


# ============================================================================
# BASIC DECORATORS
# ============================================================================

def simple_decorator(func: F) -> F:
    """Simplest decorator: wraps a function and adds behavior."""
    def wrapper(*args, **kwargs):
        print(f"  [BEFORE] Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"  [AFTER] {func.__name__} returned: {result}")
        return result
    return wrapper


def timing_decorator(func: F) -> F:
    """Decorator that measures execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  ⏱️  {func.__name__} took {elapsed*1000:.2f}ms")
        return result
    return wrapper


def logging_decorator(func: F) -> F:
    """Decorator that logs function calls with arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_str = ', '.join(repr(a) for a in args)
        kwargs_str = ', '.join(f"{k}={v!r}" for k, v in kwargs.items())
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))
        print(f"  📝 Calling {func.__name__}({all_args})")
        result = func(*args, **kwargs)
        print(f"  📝 {func.__name__} returned {result!r}")
        return result
    return wrapper


# ============================================================================
# DECORATORS WITH ARGUMENTS
# ============================================================================

def repeat_decorator(times: int) -> Callable[[F], F]:
    """Decorator factory that repeats function execution.
    
    Usage:
        @repeat_decorator(3)
        def greet(name):
            print(f"Hello, {name}!")
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                print(f"  [Iteration {i+1}/{times}]")
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator


def prefix_decorator(prefix: str) -> Callable[[F], F]:
    """Decorator that adds a prefix to print output."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_print = __builtins__.print if isinstance(__builtins__, dict) else __builtins__.print
            
            def prefixed_print(*p_args, **p_kwargs):
                p_kwargs['sep'] = p_kwargs.get('sep', ' ')
                message = p_kwargs['sep'].join(str(a) for a in p_args)
                original_print(f"{prefix}{message}", **{k: v for k, v in p_kwargs.items() if k != 'sep'})
            
            import builtins
            original = builtins.print
            builtins.print = prefixed_print
            try:
                result = func(*args, **kwargs)
            finally:
                builtins.print = original
            return result
        return wrapper
    return decorator


def rate_limit_decorator(max_calls: int, time_window: float) -> Callable[[F], F]:
    """Decorator that rate-limits function calls within a time window."""
    def decorator(func: F) -> F:
        calls = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls outside the time window
            calls[:] = [t for t in calls if now - t < time_window]
            
            if len(calls) >= max_calls:
                print(f"  ⚠️  Rate limit exceeded: max {max_calls} calls per {time_window}s")
                return None
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# MEMOIZATION AND CACHING
# ============================================================================

def memoize_decorator(func: F) -> F:
    """Simple memoization decorator that caches function results."""
    cache = {}
    calls = [0]
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a hashable key from arguments
        key = (args, tuple(sorted(kwargs.items())))
        
        if key in cache:
            print(f"  💾 Cache hit for {func.__name__}{args}")
            return cache[key]
        
        calls[0] += 1
        print(f"  🔄 Computing {func.__name__}{args} (call #{calls[0]})")
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    
    # Expose cache info
    wrapper.cache = cache
    wrapper.calls = calls
    return wrapper


def lru_cache_demo_decorator(maxsize: int = 128):
    """Custom LRU cache decorator demonstration."""
    def decorator(func: F) -> F:
        cache = {}
        access_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            
            if key in cache:
                access_order.remove(key)
                access_order.append(key)
                print(f"  💾 LRU Cache hit")
                return cache[key]
            
            if len(cache) >= maxsize:
                oldest = access_order.pop(0)
                del cache[oldest]
                print(f"  🗑️  Evicted oldest entry, cache size: {len(cache)}")
            
            result = func(*args, **kwargs)
            cache[key] = result
            access_order.append(key)
            print(f"  ✏️  Added to cache, size: {len(cache)}/{maxsize}")
            return result
        
        wrapper.cache_info = lambda: f"Size: {len(cache)}, Max: {maxsize}"
        return wrapper
    return decorator


# ============================================================================
# VALIDATION AND TYPE CHECKING
# ============================================================================

def validate_positive(func: F) -> F:
    """Decorator that validates all numeric arguments are positive."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg < 0:
                raise ValueError(f"Expected positive numbers, got {arg}")
        
        for key, val in kwargs.items():
            if isinstance(val, (int, float)) and val < 0:
                raise ValueError(f"Expected positive {key}, got {val}")
        
        return func(*args, **kwargs)
    return wrapper


def validate_types(**expected_types) -> Callable[[F], F]:
    """Decorator that validates argument types."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, param_type in expected_types.items():
                if param_name in bound.arguments:
                    actual = bound.arguments[param_name]
                    if not isinstance(actual, param_type):
                        raise TypeError(
                            f"{param_name}: expected {param_type.__name__}, "
                            f"got {type(actual).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# CLASS DECORATORS
# ============================================================================

def dataclass_like_decorator(cls):
    """Simple decorator that adds __repr__ to a class."""
    original_repr = cls.__repr__ if hasattr(cls, '__repr__') else None
    
    def new_repr(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls.__name__}({attrs})"
    
    cls.__repr__ = new_repr
    return cls


def singleton_decorator(cls):
    """Class decorator that implements the singleton pattern."""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def frozen_decorator(cls):
    """Class decorator that prevents attribute modification."""
    original_setattr = cls.__setattr__
    
    def no_setattr(self, name, value):
        if hasattr(self, '__dict__') and name in self.__dict__:
            raise AttributeError(f"Cannot modify frozen attribute {name}")
        original_setattr(self, name, value)
    
    cls.__setattr__ = no_setattr
    return cls


def debug_decorator(cls):
    """Class decorator that logs all method calls."""
    original_init_subclass = cls.__init_subclass__
    
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            def make_wrapper(original_method, method_name):
                @functools.wraps(original_method)
                def wrapper(*args, **kwargs):
                    print(f"  🔍 Calling {cls.__name__}.{method_name}")
                    return original_method(*args, **kwargs)
                return wrapper
            
            setattr(cls, attr_name, make_wrapper(attr, attr_name))
    
    return cls


# ============================================================================
# DECORATOR CHAINING
# ============================================================================

def check_auth_decorator(func: F) -> F:
    """Simulates authorization checking."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  🔐 Authorization check passed")
        return func(*args, **kwargs)
    return wrapper


def audit_decorator(func: F) -> F:
    """Simulates audit logging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  📋 Audit: {func.__name__} called")
        result = func(*args, **kwargs)
        print(f"  📋 Audit: {func.__name__} completed")
        return result
    return wrapper


# ============================================================================
# DEMO FUNCTIONS AND CLASSES
# ============================================================================

@simple_decorator
def greet(name):
    """Simple function to demonstrate basic decorator."""
    return f"Hello, {name}!"


@timing_decorator
def slow_computation(n):
    """Function to demonstrate timing decorator."""
    time.sleep(0.1)
    return sum(range(n))


@logging_decorator
def add(a, b):
    """Function to demonstrate logging decorator."""
    return a + b


@repeat_decorator(3)
def say_hello(name):
    """Function that gets repeated."""
    print(f"  Hello, {name}!")
    return name


@rate_limit_decorator(max_calls=3, time_window=1.0)
def api_call(endpoint):
    """Simulates an API call with rate limiting."""
    print(f"  📡 Calling API: {endpoint}")
    return f"Response from {endpoint}"


@memoize_decorator
def fibonacci(n):
    """Fibonacci with memoization."""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


@lru_cache_demo_decorator(maxsize=3)
def expensive_operation(x):
    """Expensive operation with LRU caching."""
    print(f"  ⚙️  Processing {x}...")
    time.sleep(0.05)
    return x ** 2


@validate_positive
def divide(a, b):
    """Division with validation."""
    return a / b


@validate_types(name=str, age=int)
def create_person(name, age):
    """Person creation with type validation."""
    return f"{name} (age {age})"


@dataclass_like_decorator
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


@audit_decorator
@check_auth_decorator
def sensitive_operation(data):
    """Operation with stacked decorators (order matters)."""
    print(f"  🔐 Processing sensitive data: {data}")
    return f"Processed: {data}"


# ============================================================================
# RUN DEMO
# ============================================================================

def run_demo():
    print("=" * 70)
    print("PYTHON DECORATORS COMPREHENSIVE DEMO")
    print("=" * 70)

    # Demo 1: Basic Decorators
    print("\n[1] BASIC DECORATORS")
    print("-" * 70)
    result = greet("Alice")
    print(f"Result: {result}\n")

    # Demo 2: Timing Decorator
    print("[2] TIMING DECORATOR")
    print("-" * 70)
    result = slow_computation(100)
    print(f"Result: {result}\n")

    # Demo 3: Logging Decorator
    print("[3] LOGGING DECORATOR")
    print("-" * 70)
    result = add(5, 3)
    print()

    # Demo 4: Decorators with Arguments
    print("[4] DECORATOR WITH ARGUMENTS (repeat_decorator)")
    print("-" * 70)
    results = say_hello("Bob")
    print()

    # Demo 5: Rate Limiting
    print("[5] RATE LIMITING DECORATOR")
    print("-" * 70)
    for i in range(5):
        api_call(f"/endpoint/{i}")
    print()

    # Demo 6: Memoization
    print("[6] MEMOIZATION DECORATOR")
    print("-" * 70)
    print("Computing fibonacci(5) with memoization:")
    result = fibonacci(5)
    print(f"Result: {result}")
    print(f"Cache info: {fibonacci.cache}")
    print()

    # Demo 7: LRU Cache
    print("[7] LRU CACHE DECORATOR")
    print("-" * 70)
    for x in [1, 2, 3, 1, 4, 2]:  # Request pattern with repeats
        result = expensive_operation(x)
        print(f"Result: {result**2}, {expensive_operation.cache_info()}")
    print()

    # Demo 8: Validation
    print("[8] VALIDATION DECORATORS")
    print("-" * 70)
    try:
        result = divide(10, 2)
        print(f"divide(10, 2) = {result}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        result = divide(-5, 2)
    except ValueError as e:
        print(f"divide(-5, 2) raised: {e}\n")

    try:
        result = create_person("Charlie", 30)
        print(f"create_person('Charlie', 30) = {result}")
    except TypeError as e:
        print(f"Error: {e}")

    try:
        create_person("David", "thirty")
    except TypeError as e:
        print(f"create_person('David', 'thirty') raised: {e}\n")

    # Demo 9: Class Decorators
    print("[9] CLASS DECORATORS")
    print("-" * 70)
    p = Person("Eve", 25)
    print(f"Person instance: {p}\n")

    # Demo 10: Decorator Chaining
    print("[10] DECORATOR CHAINING (order matters!)")
    print("-" * 70)
    sensitive_operation("confidential_data")
    print()

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("✓ Decorators modify function/class behavior without changing source")
    print("✓ @functools.wraps preserves original function metadata")
    print("✓ Decorator factories enable parametrized decorators")
    print("✓ Decorator stacking: @a @b func is equivalent to a(b(func))")
    print("✓ Common patterns: timing, logging, caching, validation, auth")
    print("✓ Class decorators can add/modify methods and attributes")
    print("✓ Composition enables powerful, reusable behavior modification")


if __name__ == "__main__":
    run_demo()
