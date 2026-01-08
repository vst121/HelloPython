import logging
import time

# Configure logging to see the internal magic happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class NonNegative:
    """
    A Data Descriptor that ensures a value is non-negative.
    
    A Data Descriptor defines both __get__ and __set__.
    Data descriptors always override instance dictionary entries.
    """
    def __init__(self, name=None):
        self.name = name

    def __set_name__(self, owner, name):
        """
        Called at class creation time. This allows the descriptor to know
        the name used in the class definition.
        """
        self.private_name = '_' + name

    def __get__(self, instance, owner):
        """
        instance: the object accessing the attribute (e.g., wallet)
        owner: the class of the object (e.g., Wallet)
        """
        if instance is None:
            return self
        return getattr(instance, self.private_name, 0)

    def __set__(self, instance, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"'{self.private_name[1:]}' must be a number.")
        if value < 0:
            raise ValueError(f"'{self.private_name[1:]}' cannot be negative.")
        
        logging.info(f"Assigning {value} to {self.private_name}")
        setattr(instance, self.private_name, value)


class LazyProperty:
    """
    A Non-Data Descriptor for lazy evaluation.
    
    A Non-Data Descriptor defines only __get__.
    Instance variables with the same name will override the descriptor's __get__.
    """
    def __init__(self, function):
        self.function = function
        self.name = function.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        logging.info(f"Computing lazy property '{self.name}'...")
        value = self.function(instance)
        
        # We 'bake' the value into the instance __dict__.
        # Subsequent lookups will find the value in __dict__ and 
        # NOT call this __get__ method again.
        setattr(instance, self.name, value)
        return value


class Wallet:
    # Data Descriptors for validation
    balance = NonNegative()
    daily_limit = NonNegative()

    def __init__(self, owner, balance, daily_limit):
        self.owner = owner
        self.balance = balance
        self.daily_limit = daily_limit

    @LazyProperty
    def transaction_history_report(self):
        """Simulate a heavy database operation or complex calculation."""
        time.sleep(2)  # Simulate delay
        return f"Report for {self.owner}: No suspicious activity detected."


def demonstrate_descriptors():
    print("--- Python Descriptors Mastery ---")
    print("Descriptors power properties, methods, and many high-level framework features.")
    
    # 1. Validation with Data Descriptors
    print("\n[1] Data Descriptor Validation:")
    try:
        my_wallet = Wallet("Alice", 1000, 200)
        print(f"Wallet for {my_wallet.owner} initialized with {my_wallet.balance}$.")
        
        print("\nAttempting to set a negative balance...")
        my_wallet.balance = -50
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        print("\nAttempting to set a non-numeric daily limit...")
        my_wallet.daily_limit = "unlimited"
    except TypeError as e:
        print(f"Caught expected error: {e}")

    # 2. Resource Management with Non-Data Descriptors
    print("\n[2] Non-Data Descriptor (Lazy Loading):")
    print("Accessing 'transaction_history_report' for the FIRST time (should be slow):")
    start = time.time()
    report1 = my_wallet.transaction_history_report
    print(f"Output: {report1}")
    print(f"Time taken: {time.time() - start:.4f} seconds")

    print("\nAccessing 'transaction_history_report' AGAIN (should be cached):")
    start = time.time()
    report2 = my_wallet.transaction_history_report
    print(f"Output: {report2}")
    print(f"Time taken: {time.time() - start:.4f} seconds")

    # 3. Inspecting the instance dictionary
    print("\n[3] Introspection:")
    print(f"Wallet attributes in __dict__: {list(my_wallet.__dict__.keys())}")
    print("Notice how 'balance' stores its value in '_balance' while 'transaction_history_report' is cached directly.")

if __name__ == "__main__":
    demonstrate_descriptors()
    print("\nSummary:")
    print("- Data Descriptors: Define __set__ (and likely __get__). Useful for validation/enforcement.")
    print("- Non-Data Descriptors: Define only __get__. Useful for performance hacks like lazy-caching.")
    print("- __set_name__: Simplifies attribute naming (Python 3.6+).")
