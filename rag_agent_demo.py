"""
rag_agent_demo.py

Demo: RAG (Retrieval-Augmented Generation) Agent.

This agent:
1. RETRIEVES: Searches a knowledge base for relevant documents
2. AUGMENTS: Enriches prompt with retrieved context
3. GENERATES: Uses LLM to create answer based on retrieved info
4. CITES: References sources for transparency and fact-checking
5. LEARNS: Optionally stores Q&A pairs for future retrieval

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python rag_agent_demo.py

This pattern powers ChatGPT's knowledge retrieval, Retrieval Augmented Generation,
and is essential for accurate, source-aware AI systems. Used by Anthropic, OpenAI,
and most production LLM applications.
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import re

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
# Document & Knowledge Base
# ============================================================================

@dataclass
class Document:
    """Represents a document in the knowledge base."""
    doc_id: str
    title: str
    content: str
    source: str
    metadata: Dict[str, Any] = None

    def __repr__(self):
        return f"Document({self.doc_id}, {self.title[:30]}...)"


class KnowledgeBase:
    """Simple in-memory knowledge base with semantic search (basic version)."""

    def __init__(self):
        self.documents: List[Document] = []
        self.doc_by_id: Dict[str, Document] = {}

    def add_document(
        self, doc_id: str, title: str, content: str, source: str, metadata: Dict = None
    ) -> Document:
        """Add a document to the knowledge base."""
        doc = Document(doc_id, title, content, source, metadata or {})
        self.documents.append(doc)
        self.doc_by_id[doc_id] = doc
        return doc

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Search knowledge base (simple keyword-based for demo)."""
        query_words = set(query.lower().split())
        results = []

        for doc in self.documents:
            # Simple relevance: count matching keywords
            content_lower = doc.content.lower()
            title_lower = doc.title.lower()

            score = 0
            for word in query_words:
                score += content_lower.count(word) * 1.0
                score += title_lower.count(word) * 2.0  # Title matches weighted higher

            if score > 0:
                results.append((doc, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def __len__(self):
        return len(self.documents)


# ============================================================================
# Sample Knowledge Base
# ============================================================================

def create_sample_kb() -> KnowledgeBase:
    """Create a sample knowledge base."""
    kb = KnowledgeBase()

    # Add sample documents
    kb.add_document(
        doc_id="python_basics_1",
        title="Python Basics: Variables and Data Types",
        content="""Python is a high-level, dynamically-typed language. Variables are created
by assignment. Common data types include: int, float, str, list, dict, tuple, set.
Python uses indentation for code blocks. Everything in Python is an object.""",
        source="Python Documentation",
        metadata={"category": "python", "level": "beginner"},
    )

    kb.add_document(
        doc_id="python_functions",
        title="Python Functions and Decorators",
        content="""Functions in Python are defined with the def keyword. They can have default
arguments, *args, and **kwargs. Decorators are functions that modify other functions.
Lambda functions are anonymous functions. Function scope follows LEGB rule.""",
        source="Python Documentation",
        metadata={"category": "python", "level": "intermediate"},
    )

    kb.add_document(
        doc_id="ai_agents",
        title="AI Agent Patterns and Architectures",
        content="""AI agents are autonomous systems that perceive their environment and take
actions. Common patterns: ReAct (Reasoning + Acting), RAG (Retrieval-Augmented),
Multi-Agent systems, Autonomous agents with memory. Agents need tools, reasoning,
and feedback loops for effective operation.""",
        source="AI Research Papers",
        metadata={"category": "ai", "level": "advanced"},
    )

    kb.add_document(
        doc_id="llm_best_practices",
        title="LLM Best Practices and Prompt Engineering",
        content="""Effective LLM usage requires: clear prompts, few-shot examples, temperature
tuning, token limits, and context management. Chain-of-Thought prompting improves
reasoning. Prompt injection risks require input validation. Always include system
prompts to guide behavior.""",
        source="LLM Guidelines",
        metadata={"category": "llm", "level": "intermediate"},
    )

    kb.add_document(
        doc_id="rag_systems",
        title="RAG Systems: Retrieval-Augmented Generation",
        content="""RAG combines retrieval and generation: retrieve relevant documents, augment
the prompt with retrieved context, then generate the answer. RAG improves accuracy,
enables source attribution, reduces hallucination, and supports knowledge updates.
Used in production by OpenAI, Anthropic, and major AI companies.""",
        source="RAG Research",
        metadata={"category": "rag", "level": "advanced"},
    )

    return kb


# ============================================================================
# RAG Agent
# ============================================================================

class RAGAgent:
    """
    RAG Agent: Retrieves relevant documents and augments generation with context.
    This is one of the most popular patterns in production LLM systems.
    """

    def __init__(self, kb: KnowledgeBase, llm_provider: str = "ollama"):
        self.kb = kb
        self.llm_provider = llm_provider
        self.qa_history: List[Dict[str, Any]] = []
        self.retrieval_threshold = 0.0  # Any positive score is relevant

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve relevant documents from knowledge base."""
        results = self.kb.search(query, top_k=top_k)
        documents = [doc for doc, score in results]
        return documents

    def augment_prompt(
        self, query: str, documents: List[Document]
    ) -> Tuple[str, str]:
        """Create augmented prompt with retrieved context."""
        context = ""
        for i, doc in enumerate(documents, 1):
            context += f"\n[Source {i}: {doc.title}]\n{doc.content}\n"

        augmented_prompt = f"""Based on the following retrieved documents, answer the question.
Cite your sources by referencing [Source N].

RETRIEVED CONTEXT:{context}

QUESTION: {query}

ANSWER:"""

        return augmented_prompt, context

    def generate(self, augmented_prompt: str) -> str:
        """Generate answer using LLM with augmented prompt."""
        system_prompt = """You are a helpful assistant answering questions based on provided context.
Always cite your sources using [Source N] format. If information is not in the context, say so.
Be accurate and concise."""

        response = call_llm(augmented_prompt, self.llm_provider, system_prompt)
        return response

    def query(self, question: str, top_k: int = 3, include_reasoning: bool = True) -> Dict[str, Any]:
        """Full RAG pipeline: Retrieve → Augment → Generate."""
        # Step 1: Retrieve
        documents = self.retrieve(question, top_k=top_k)

        # Step 2: Augment
        augmented_prompt, context = self.augment_prompt(question, documents)

        # Step 3: Generate
        answer = self.generate(augmented_prompt)

        # Store in history
        qa_record = {
            "query": question,
            "documents_retrieved": [d.title for d in documents],
            "answer": answer,
            "source_count": len(documents),
        }
        self.qa_history.append(qa_record)

        result = {
            "question": question,
            "retrieved_docs": [
                {
                    "title": doc.title,
                    "source": doc.source,
                    "doc_id": doc.doc_id,
                }
                for doc in documents
            ],
            "answer": answer,
            "success": True,
        }

        if include_reasoning:
            result["augmented_prompt"] = augmented_prompt[:300]

        return result

    def run_interactive(self):
        """Run interactive RAG agent."""
        print(f"\n{'='*70}")
        print(f"RAG Agent (using {self.llm_provider})")
        print(f"{'='*70}\n")

        print(f"Knowledge Base loaded: {len(self.kb)} documents\n")

        example_queries = [
            "What are Python decorators?",
            "Tell me about AI agents",
            "What is RAG?",
        ]

        for i, query in enumerate(example_queries, 1):
            print(f"{'='*70}")
            print(f"Query {i}: {query}")
            print(f"{'='*70}\n")

            # Retrieve
            docs = self.retrieve(query, top_k=2)
            print("📚 Retrieved Documents:")
            for j, doc in enumerate(docs, 1):
                print(f"  {j}. {doc.title} (Source: {doc.source})")

            # Augment and Generate
            augmented_prompt, _ = self.augment_prompt(query, docs)
            print("\n💡 Generating answer with LLM...")

            answer = self.generate(augmented_prompt)
            print(f"\n✓ Answer:\n{answer}\n")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "kb_size": len(self.kb),
            "queries_processed": len(self.qa_history),
            "avg_sources_per_query": (
                sum(qa["source_count"] for qa in self.qa_history) / len(self.qa_history)
                if self.qa_history
                else 0
            ),
        }


# ============================================================================
# LLM Integration
# ============================================================================

def call_llm(prompt: str, provider: str = "ollama", system_prompt: str = None) -> str:
    """Call LLM with a prompt."""
    if provider.lower() == "openai":
        return call_openai(prompt, system_prompt)
    else:
        return call_ollama(prompt, system_prompt)


def call_ollama(prompt: str, system_prompt: str = None) -> str:
    """Call Ollama model."""
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
        resp = requests.post(
            url,
            json={"messages": messages},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                return msg.get("content", "").strip()

        return str(data)
    except Exception as e:
        return f"(Error: {e})"


def call_openai(prompt: str, system_prompt: str = None) -> str:
    """Call OpenAI model."""
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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Main - Demo
# ============================================================================

if __name__ == "__main__":
    print("RAG Agent Demo")
    print("=" * 70)

    # Create knowledge base
    print("\n📚 Creating knowledge base...")
    kb = create_sample_kb()
    print(f"✓ Knowledge base loaded with {len(kb)} documents\n")

    # Create RAG agent
    agent = RAGAgent(kb, llm_provider="ollama")

    print("RAG Pipeline:")
    print("1. RETRIEVE: Search knowledge base for relevant documents")
    print("2. AUGMENT: Add retrieved context to prompt")
    print("3. GENERATE: Use LLM to answer with context")
    print("4. CITE: Reference sources in answer\n")

    print("Note: This demo shows the RAG pattern.")
    print("Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real LLM execution.\n")

    # Example queries
    print("Example Queries:")
    queries = [
        "What are Python decorators?",
        "Explain RAG systems",
        "Tell me about AI agents",
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n{i}. {q}")
        docs = agent.retrieve(q, top_k=2)
        print(f"   Retrieved documents: {', '.join([d.title[:40] for d in docs])}")

    # Stats
    print(f"\n{'='*70}")
    print("Agent Statistics:")
    stats = agent.get_stats()
    print(f"  Knowledge base size: {stats['kb_size']} documents")
    print(f"  This pattern is used by: OpenAI, Anthropic, Major LLM companies")

    # Uncomment to run with real LLM:
    # agent.run_interactive()
