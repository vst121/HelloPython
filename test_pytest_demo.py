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

    def transfer(self, target_account, amount):
        """
        Integration Point: Interaction between two BankAccount objects.
        """
        self.withdraw(amount)
        target_account.deposit(amount)

    def get_market_conversion_rate(self, currency):
        """
        Simulates a slow, external API call to get currency rates.
        We will mock this in our tests.
        """
        print(f"  [API] Fetching rate for {currency}...")
        time.sleep(2)  # Simulate network latency
        return 1.1  # Hardcoded for simulation

class TransactionLogger:
    """
    Simulates a service that interacts with the filesystem.
    This will be used for Integration Testing.
    """
    def __init__(self, log_file):
        self.log_file = log_file

    def log_transaction(self, message):
        with open(self.log_file, "a") as f:
            f.write(f"{time.ctime()}: {message}\n")

    def get_logs(self):
        try:
            with open(self.log_file, "r") as f:
                return f.readlines()
        except FileNotFoundError:
            return []

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

# 6. Integration Testing: Interaction between Multiple Components
def test_account_transfer_integration(funded_account, empty_account):
    """
    Tests the interaction between two BankAccount instances.
    """
    initial_funded_balance = funded_account.balance
    initial_empty_balance = empty_account.balance
    transfer_amount = 50

    funded_account.transfer(empty_account, transfer_amount)

    assert funded_account.balance == initial_funded_balance - transfer_amount
    assert empty_account.balance == initial_empty_balance + transfer_amount

def test_transaction_logger_integration(tmp_path):
    """
    Tests integration with the filesystem using pytest's built-in 'tmp_path' fixture.
    This verifies that the TransactionLogger correctly writes and reads from a real file.
    """
    # tmp_path is a pathlib.Path object pointing to a temporary directory unique to the test
    log_file = tmp_path / "test_log.txt"
    logger = TransactionLogger(log_file)

    logger.log_transaction("Deposit of 100")
    logger.log_transaction("Withdrawal of 50")

    logs = logger.get_logs()
    
    assert len(logs) == 2
    assert "Deposit of 100" in logs[0]
    assert "Withdrawal of 50" in logs[1]
    assert log_file.exists()  # Verify the file was actually created on disk

# 6. Built-in help: Showing how to run this file directly
if __name__ == "__main__":
    print("\nRunning Pytest Demonstration...")
    print("=" * 60)
    # This allows running the file like 'python test_pytest_demo.py'
    # which will trigger pytest to run the tests in this file.
    pytest.main([__file__])
