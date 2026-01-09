import pytest
import time

# --- Code to be tested ---

class BankAccount:
    """
    A simple BankAccount class to demonstrate pytest testing.
    """
    def __init__(self, initial_balance=0):
        self.balance = initial_balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        self.balance -= amount
        return self.balance

    def get_market_conversion_rate(self, currency):
        """
        Simulates a slow, external API call to get currency rates.
        We will mock this in our tests.
        """
        print(f"  [API] Fetching rate for {currency}...")
        time.sleep(2)  # Simulate network latency
        return 1.1  # Hardcoded for simulation

# --- Pytest Tests ---

# 1. Fixtures: Reusable setup for tests
@pytest.fixture
def empty_account():
    """Returns a BankAccount with 0 balance."""
    return BankAccount(0)

@pytest.fixture
def funded_account():
    """Returns a BankAccount with 100 balance."""
    return BankAccount(100)

# 2. Basic Assertion Tests
def test_new_account_balance(empty_account):
    assert empty_account.balance == 0

def test_deposit(empty_account):
    empty_account.deposit(50)
    assert empty_account.balance == 50

# 3. Exception Testing: Ensuring the right errors are raised
def test_withdraw_insufficient_funds(funded_account):
    with pytest.raises(ValueError, match="Insufficient funds"):
        funded_account.withdraw(150)

def test_negative_deposit(empty_account):
    with pytest.raises(ValueError, match="Deposit amount must be positive"):
        empty_account.deposit(-10)

# 4. Parametrization: Running the same test logic with different inputs
@pytest.mark.parametrize("deposit_amount, expected_balance", [
    (10, 110),
    (20.5, 120.5),
    (100, 200),
])
def test_multiple_deposits(funded_account, deposit_amount, expected_balance):
    funded_account.deposit(deposit_amount)
    assert funded_account.balance == expected_balance

# 5. Mocking / Monkeypatching: Overriding external dependencies
def test_conversion_rate_mocked(monkeypatch, funded_account):
    """
    Demonstrates monkeypatching a method to avoid a slow API call.
    """
    def mock_get_rate(self, currency):
        return 1.5  # Fixed rate for testing, no sleep!

    # Replace the real method with our mock
    monkeypatch.setattr(BankAccount, "get_market_conversion_rate", mock_get_rate)

    rate = funded_account.get_market_conversion_rate("EUR")
    assert rate == 1.5
    assert funded_account.balance == 100  # Balance remains unchanged

# 6. Built-in help: Showing how to run this file directly
if __name__ == "__main__":
    print("\nRunning Pytest Demonstration...")
    print("=" * 60)
    # This allows running the file like 'python test_pytest_demo.py'
    # which will trigger pytest to run the tests in this file.
    pytest.main([__file__])
