def simple_closure_demo():
    """A basic example of a closure."""
    def outer_function(msg):
        # msg is a local variable to outer_function
        def inner_function():
            # inner_function 'closes over' the msg variable
            print(f"Message from closure: {msg}")
        
        return inner_function

    my_func = outer_function("Hello from the closure world!")
    # Even though outer_function has finished execution,
    # inner_function still has access to 'msg'.
    my_func()


def stateful_closure_demo():
    """Example of using closures to maintain state without using a class."""
    def make_counter():
        count = 0  # Initial state
        
        def counter():
            nonlocal count  # Allows modification of the variable in the outer scope
            count += 1
            return count
        
        return counter

    counter_a = make_counter()
    print(f"Counter A: {counter_a()}")
    print(f"Counter A: {counter_a()}")

    counter_b = make_counter()  # A completely separate state
    print(f"Counter B: {counter_b()}")
    print(f"Counter A: {counter_a()}")


def function_factory_demo():
    """Example of a function factory using closures."""
    def power_factory(exponent):
        def power(base):
            return base ** exponent
        return power

    square = power_factory(2)
    cube = power_factory(3)

    print(f"Square of 5: {square(5)}")
    print(f"Cube of 5: {cube(5)}")


def run_demo():
    """Main demonstration logic."""
    print("Demonstrating Python Closures:")
    print("=" * 60)

    print("1. Simple Closure Example:")
    simple_closure_demo()
    print()

    print("2. Stateful Closure (Counter):")
    stateful_closure_demo()
    print()

    print("3. Function Factory Example:")
    function_factory_demo()
    print()

    print("Key Takeaways:")
    print("- A closure is a function object that remembers values in enclosing scopes.")
    print("- Closures can be used to avoid global variables and provide data hiding.")
    print("- The 'nonlocal' keyword is used to modify variables in the outer (non-global) scope.")
    print("- They are a lightweight alternative to classes for simple state management.")

if __name__ == "__main__":
    run_demo()

# Python Closures Summary:
# A closure occurs when a nested function references a value in its enclosing scope.
# The closure "closes over" the free variables, keeping them alive even after the 
# outer function has returned. This is the underlying mechanism for decorators.
