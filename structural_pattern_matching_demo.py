def describe_point(point):
    """
    1. Literal and Sequence Matching. 
    Matches tuples or lists of specific shapes.
    """
    match point:
        case (0, 0):
            return "Origin"
        case (0, y):
            return f"On the Y-axis at y={y}"
        case (x, 0):
            return f"On the X-axis at x={x}"
        case (x, y) if x == y:
            return f"On the line y=x at {x}"
        case (x, y):
            return f"At point ({x}, {y})"
        case _:
            return "Not a valid 2D point"

def process_command(command):
    """
    2. Complex Sequence and Mapping Matching.
    Matches commands with optional flags or sub-commands.
    """
    match command.split():
        case ["quit"]:
            return "Executing system exit..."
        case ["load", filename]:
            return f"Loading file: {filename}"
        case ["save", filename, ("--force" | "-f")]:
            return f"Forcefully saving file: {filename}"
        case ["save", filename]:
            return f"Saving file: {filename}"
        case ["move", ("up" | "down" | "left" | "right") as direction, steps]:
            return f"Moving {direction} by {steps} units"
        case _:
            return "Unknown command"

class User:
    def __init__(self, name, role, active=True):
        self.name = name
        self.role = role
        self.active = active

def handle_user(user):
    """
    3. Class/Object Matching.
    Unpacks object attributes directly in the pattern.
    """
    match user:
        case User(name=name, role="Admin", active=True):
            return f"Welcome, Administrator {name}! Full access granted."
        case User(name=name, role="Editor", active=True):
            return f"Hello {name}, you can edit content."
        case User(active=False):
            return "Account is disabled."
        case User(name=name):
            return f"Guest {name} has limited access."
        case _:
            return "Unrecognized object type"

def handle_api_response(response):
    """
    4. Mapping (Dict) Matching.
    Matches specific keys and values in dictionaries.
    """
    match response:
        case {"status": 200, "data": data}:
            return f"Success! Received: {data}"
        case {"status": 404}:
            return "Error: Resource not found."
        case {"status": code, "error": message}:
            return f"Error {code}: {message}"
        case _:
            return "Invalid response format"

def demonstrate_matching():
    print("--- Structural Pattern Matching Mastery (Python 3.10+) ---")
    print("Higher-level than a switch-case; allows destructuring and validation.")

    # 1. Sequence Matching
    print("\n[1] Sequence/Coordinate Matching:")
    points = [(0, 0), (0, 10), (5, 5), (3, 4), "junk"]
    for p in points:
        print(f"  {p} -> {describe_point(p)}")

    # 2. Command Processing
    print("\n[2] Shell Command Matching:")
    commands = ["load data.csv", "save backup.db --force", "move up 10", "quit", "help"]
    for cmd in commands:
        print(f"  '{cmd}' -> {process_command(cmd)}")

    # 3. Object Matching
    print("\n[3] Object/Class Matching:")
    users = [
        User("Alice", "Admin"),
        User("Bob", "Editor", active=False),
        User("Charlie", "Guest")
    ]
    for u in users:
        print(f"  User({u.name}, {u.role}) -> {handle_user(u)}")

    # 4. Dictionary Matching
    print("\n[4] Dictionary/API Matching:")
    responses = [
        {"status": 200, "data": "User profile info"},
        {"status": 404},
        {"status": 500, "error": "Internal Server Error"}
    ]
    for resp in responses:
        print(f"  {resp} -> {handle_api_response(resp)}")

if __name__ == "__main__":
    demonstrate_matching()
    print("\nSummary:")
    print("- Use 'match-case' for complex conditional logic and data destructuring.")
    print("- Sequence matching works for lists/tuples.")
    print("- Class matching looks at object attributes.")
    print("- Use 'as' to bind parts of a pattern to a variable.")
    print("- Guards ('if') allow for additional runtime checks within a case.")
