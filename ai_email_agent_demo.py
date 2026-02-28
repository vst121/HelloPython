"""
ai_email_agent_demo.py

Demo: AI Agent that reads emails and summarizes them.

This agent:
1. Fetches emails (mock or IMAP)
2. Uses an LLM (Ollama/phi3 or OpenAI) to summarize each
3. Optionally groups summaries by topic

Usage:
- pip install requests (for Ollama) or openai (for GPT)
- Set env vars: OLLAMA_HOST, OLLAMA_MODEL, or OPENAI_API_KEY
- python ai_email_agent_demo.py

Features:
- Demonstrates agent-like behavior (read → analyze → summarize)
- Fallback to mock emails if Ollama/OpenAI unavailable
- Multi-turn agent loop
"""

import os
import json
import re
from typing import Any

# Try to import LLM clients
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


# ============================================================================
# Email Sources (Mock or Real)
# ============================================================================

MOCK_EMAILS = [
    {
        "from": "alice@example.com",
        "subject": "Q1 Budget Review",
        "body": "Please review the attached Q1 budget proposal. We need to approve by Friday. "
                "Key items: 15% increase in marketing spend, new headcount for engineering, "
                "reduced travel budget due to remote work expansion.",
    },
    {
        "from": "bob@company.com",
        "subject": "Urgent: Server Downtime Issue",
        "body": "Production database went down at 2 PM UTC. Root cause: disk space exhaustion. "
                "Team is restoring from backup now. All systems should be online within 30 mins. "
                "Post-mortem meeting scheduled for tomorrow.",
    },
    {
        "from": "carol@company.com",
        "subject": "New Team Member Onboarding",
        "body": "Please welcome Sarah, our new Data Scientist joining the Analytics team. "
                "Her first day is Monday. She has experience with Python, SQL, and Tableau. "
                "Can someone prepare a dev environment and send her the onboarding docs?",
    },
]


def get_emails(use_mock=True):
    """Return a list of email dicts with from/subject/body."""
    if use_mock:
        return MOCK_EMAILS
    # Real IMAP fetch would go here (left as exercise)
    return []


# ============================================================================
# LLM Integration (Ollama or OpenAI)
# ============================================================================

def summarize_via_ollama(text: str, host: str = None, model: str = None) -> str:
    """Summarize text using a local Ollama model."""
    if not OLLAMA_AVAILABLE:
        return "(Ollama not available)"

    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "phi3")
    url = f"{host.rstrip('/')}/chat?model={model}"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Summarize the following email in 2-3 sentences.",
        },
        {"role": "user", "content": f"Email body:\n{text}"},
    ]

    try:
        resp = requests.post(
            url,
            json={"messages": messages},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract content from response
        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                return msg.get("content", "").strip()

        return str(data)
    except Exception as e:
        return f"(Ollama error: {e})"


def summarize_via_openai(text: str, api_key: str = None) -> str:
    """Summarize text using OpenAI GPT."""
    if not OPENAI_AVAILABLE:
        return "(OpenAI not available)"

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(No OpenAI API key)"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following email in 2-3 sentences.",
                },
                {"role": "user", "content": f"Email body:\n{text}"},
            ],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error: {e})"


def summarize_email(subject: str, body: str, provider: str = "ollama") -> str:
    """Pick the best available LLM and summarize."""
    if provider.lower() == "openai":
        summary = summarize_via_openai(body)
    else:
        summary = summarize_via_ollama(body)

    return summary


# ============================================================================
# Email Agent
# ============================================================================

class EmailSummarizerAgent:
    """Simple agent that reads and summarizes emails."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.summaries = []

    def process_emails(self, emails: list) -> list:
        """Process each email and return list of summaries."""
        print(f"\n{'='*70}")
        print(f"Email Summarizer Agent (using {self.llm_provider})")
        print(f"{'='*70}\n")

        for i, email in enumerate(emails, 1):
            print(f"[{i}/{len(emails)}] From: {email['from']}")
            print(f"Subject: {email['subject']}\n")

            summary = summarize_email(email["subject"], email["body"], self.llm_provider)
            print(f"Summary: {summary}\n")

            self.summaries.append(
                {
                    "from": email["from"],
                    "subject": email["subject"],
                    "summary": summary,
                }
            )

        return self.summaries

    def get_summary_report(self) -> str:
        """Generate a brief report of all summaries."""
        report = "\n" + "=" * 70 + "\n"
        report += "SUMMARY REPORT\n"
        report += "=" * 70 + "\n"

        for item in self.summaries:
            report += f"\n[{item['from']}] {item['subject']}\n"
            report += f"  → {item['summary']}\n"

        return report


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Create agent
    agent = EmailSummarizerAgent(llm_provider="ollama")

    # Fetch emails (mock by default)
    emails = get_emails(use_mock=True)

    # Process emails
    agent.process_emails(emails)

    # Print report
    print(agent.get_summary_report())

    print("\nAgent completed. Set OLLAMA_HOST and OLLAMA_MODEL env vars to use a real Ollama instance.")
