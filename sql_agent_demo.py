"""
sql_agent_demo.py

Demo: SQL Query Agent – translates natural language to SQL, executes
against a database, and returns results. This is extremely popular in
analytics tools (e.g. GitHub Copilot for SQL, AI in BI platforms like
Looker, Tableau, Metabase, etc.).

Flow:
1. User asks a question about the data
2. Agent prompts LLM to write a SQL query
3. Run query on SQLite in-memory mock dataset
4. Return results or error messages
5. Optionally ask the LLM to refine query on failure

Database is simulated with sample tables for demo purposes.

Usage:
- pip install requests sqlite3
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python sql_agent_demo.py

"""

import os
import sqlite3
from typing import List, Tuple

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mock database setup
# ---------------------------------------------------------------------------

def create_demo_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("""CREATE TABLE employees (
                 id INTEGER PRIMARY KEY,
                 name TEXT,
                 department TEXT,
                 salary REAL)
              """)
    c.executemany(
        "INSERT INTO employees (name, department, salary) VALUES (?,?,?)",
        [
            ("Alice", "Engineering", 120000),
            ("Bob", "HR", 80000),
            ("Carol", "Engineering", 115000),
            ("Dave", "Sales", 95000),
            ("Eve", "Engineering", 105000),
        ],
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class SQLAgent:
    """Natural language -> SQL agent with execution."""

    def __init__(self, conn: sqlite3.Connection, llm_provider: str = "ollama"):
        self.conn = conn
        self.llm_provider = llm_provider

    def nl_to_sql(self, question: str) -> str:
        """Ask LLM to write a SQL query matching the question."""
        prompt = (
            "Translate the following natural language question into a valid SQL query."
            " Use the 'employees' table defined as (id,name,department,salary)."
            f"\n\nQuestion: {question}\n\nSQL:"
        )
        return call_llm(prompt, self.llm_provider)

    def execute_query(self, query: str) -> Tuple[List[Tuple], List[str]]:
        """Run SQL and return rows plus column names."""
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            cols = [description[0] for description in cur.description]
            rows = cur.fetchall()
            return rows, cols
        except Exception as e:
            return [], [str(e)]

    def ask(self, question: str) -> None:
        print(f"\nUser question: {question}")
        sql = self.nl_to_sql(question)
        print(f"Generated SQL:\n{sql}\n")

        rows, cols = self.execute_query(sql)
        if rows:
            print("Results:")
            print(cols)
            for r in rows:
                print(r)
        else:
            print("Error or no results:")
            print(cols)


# ---------------------------------------------------------------------------
# LLM helpers (same pattern as other demos)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, provider: str = "ollama", system_prompt: str = None) -> str:
    if provider.lower() == "openai":
        return call_openai(prompt, system_prompt)
    else:
        return call_ollama(prompt, system_prompt)


def call_ollama(prompt: str, system_prompt: str = None) -> str:
    if not OLLAMA_AVAILABLE:
        return "(Ollama not available)"
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3")
    url = f"{host.rstrip('/')}/chat?model={model}"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(url, json={"messages": messages}, headers={"Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                return choices[0].get("message", {}).get("content", "").strip()
        return str(data)
    except Exception as e:
        return f"(Error: {e})"


def call_openai(prompt: str, system_prompt: str = None) -> str:
    if not OPENAI_AVAILABLE:
        return "(OpenAI not available)"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(No API key)"
    try:
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1000)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Demo run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn = create_demo_db()
    agent = SQLAgent(conn, llm_provider="ollama")

    questions = [
        "What are the names of employees in the Engineering department?",
        "Who has a salary greater than 100000?",
        "How many employees work in HR?",
    ]

    for q in questions:
        agent.ask(q)
