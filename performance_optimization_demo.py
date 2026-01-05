import timeit
import time
from functools import lru_cache

# 1. Using __slots__ for Memory and Speed
class StandardPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedPoint:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

def slots_demo():
    """Demonstrates how __slots__ can improve memory and attribute access speed."""
    print("1. __slots__ Optimization:")
    
    # Timing attribute access
    std_points = [StandardPoint(i, i) for i in range(1000)]
    slot_points = [SlottedPoint(i, i) for i in range(1000)]
    
    def access_std():
        for p in std_points:
            z = p.x + p.y
            
    def access_slot():
        for p in slot_points:
            z = p.x + p.y
            
    std_time = timeit.timeit(access_std, number=1000)
    slot_time = timeit.timeit(access_slot, number=1000)
    
    print(f"   Standard Class Access: {std_time:.4f}s")
    print(f"   Slotted Class Access:  {slot_time:.4f}s")
    print(f"   Improvement: {((std_time - slot_time) / std_time) * 100:.1f}%")


# 2. LRU Cache for Memoization
@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n < 2: return n
    return fibonacci_cached(n-1) + fibonacci_cached(n-2)

def fibonacci_no_cache(n):
    if n < 2: return n
    return fibonacci_no_cache(n-1) + fibonacci_no_cache(n-2)

def memoization_demo():
    """Demonstrates the power of functools.lru_cache."""
    print("\n2. Memoization with lru_cache:")
    
    # Recursive fib(30) is slow without caching
    start = time.time()
    fib_val = fibonacci_cached(35)
    end = time.time()
    cached_time = end - start
    print(f"   Cached Fibonacci(35): {fib_val} (Time: {cached_time:.6f}s)")
    
    # Don't run no_cache for 35, it's too slow, let's just note it
    print("   Non-cached Fibonacci for n=35 would take millions of redundant calls.")


# 3. List Comprehensions vs Loops
def list_comp_demo():
    """Compares the performance of list comprehensions vs traditional for loops."""
    print("\n3. List Comprehensions vs Loops:")
    
    def manual_loop():
        res = []
        for i in range(10000):
            res.append(i * 2)
        return res
        
    def list_comp():
        return [i * 2 for i in range(10000)]
        
    loop_time = timeit.timeit(manual_loop, number=100)
    comp_time = timeit.timeit(list_comp, number=100)
    
    print(f"   Manual Loop Time:       {loop_time:.4f}s")
    print(f"   List Comprehension:     {comp_time:.4f}s")
    print(f"   Improvement: {((loop_time - comp_time) / loop_time) * 100:.1f}%")


# 4. Local vs Global Variable Access
GLOBAL_VAL = 100

def access_efficiency_demo():
    """Demonstrates that local variable access is faster than global."""
    print("\n4. Local vs Global Lookups:")
    
    def use_global():
        total = 0
        for i in range(1000000):
            total += GLOBAL_VAL
        return total
        
    def use_local():
        local_val = GLOBAL_VAL # Cache global into local
        total = 0
        for i in range(1000000):
            total += local_val
        return total
        
    global_time = timeit.timeit(use_global, number=10)
    local_time = timeit.timeit(use_local, number=10)
    
    print(f"   Global Lookup Time:     {global_time:.4f}s")
    print(f"   Local Lookup Time:      {local_time:.4f}s")
    print(f"   Improvement: {((global_time - local_time) / global_time) * 100:.1f}%")


def run_demo():
    """Main demonstration logic."""
    print("Demonstrating Python Performance Optimization:")
    print("=" * 60)

    slots_demo()
    memoization_demo()
    list_comp_demo()
    access_efficiency_demo()

    print("\nKey Takeaways:")
    print("- Use '__slots__' in classes to save memory and speed up attribute access.")
    print("- Use 'functools.lru_cache' to optimize recursive or repetitive functions.")
    print("- Prefer list comprehensions and built-in functions over manual loops.")
    print("- Accessing local variables is faster than global ones in Python.")

if __name__ == "__main__":
    run_demo()

# Performance Optimization Summary:
# Python is high-level, but small changes can lead to significant speedups.
# Benchmarking (using timeit) is crucial for identifying bottlenecks.
# Optimization should be targeted and only applied where performance is critical.
