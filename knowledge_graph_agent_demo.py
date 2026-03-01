"""
knowledge_graph_agent_demo.py

Demo: Knowledge Graph Agent – extracts entities and relations from text
and builds a simple in-memory graph. The agent can then answer
questions by traversing the graph. Knowledge graphs power systems like
Wikidata, Google's Knowledge Graph, and many enterprise search tools.

Flow:
1. Ingest text (documents, conversation) and use LLM to identify
   entities and their relationships
2. Store them in a graph (nodes=entities, edges=relations)
3. Accept natural language queries that are translated to graph
   traversals
4. Return answers using evidence from the graph

This pattern is popular in question-answering, semantic search, and
explainable AI. Combining LLM extraction with structured graph queries
brings together generative and symbolic reasoning.

Usage:
- pip install requests networkx
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python knowledge_graph_agent_demo.py

"""

import os
from typing import List, Tuple, Dict

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

try:
    import networkx as nx
except ImportError:
    nx = None


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def add_relation(graph, subj: str, rel: str, obj: str):
    if not graph.has_node(subj):
        graph.add_node(subj)
    if not graph.has_node(obj):
        graph.add_node(obj)
    graph.add_edge(subj, obj, relation=rel)


def query_graph(graph, start: str, relation: str) -> List[str]:
    results = []
    for u, v, data in graph.edges(data=True):
        if u == start and data.get('relation') == relation:
            results.append(v)
    return results


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class KnowledgeGraphAgent:
    """Aggregates information into a knowledge graph and answers queries."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.graph = nx.DiGraph() if nx else None

    def extract_entities_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Use LLM to extract triples (subject, relation, object)."""
        prompt = (
            "Extract all entity relationships from the text below in the "
            "format: SUBJECT | RELATION | OBJECT (one per line). Only return "
            "triples.\n\nText:\n" + text + "\n\nTriples:"
        )
        response = call_llm(prompt, self.llm_provider)
        triples = []
        for line in response.split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
        return triples

    def ingest(self, text: str):
        triples = self.extract_entities_relations(text)
        for subj, rel, obj in triples:
            add_relation(self.graph, subj, rel, obj)
        print(f"Ingested {len(triples)} triples into graph.")

    def answer_query(self, question: str) -> str:
        # simple heuristic: ask LLM to map question to graph traversal
        prompt = (
            "Given the following question and a knowledge graph of entities, "
            "output a traversal specification in the form: START | RELATION. "
            "Then return all matching objects from the graph.\n\n"
            f"Question: {question}\n\nGraph nodes: {list(self.graph.nodes())}\n"
            + "\nTraversal:"
        )
        traversal = call_llm(prompt, self.llm_provider)
        parts = [p.strip() for p in traversal.split("|")]
        if len(parts) == 2:
            start, rel = parts
            objs = query_graph(self.graph, start, rel)
            return f"Objects reachable: {objs}"
        return "Could not interpret traversal."


# ---------------------------------------------------------------------------
# LLM helpers (same as other demos)
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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1500)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not nx:
        print("NetworkX is required for the knowledge graph. Please install it.")
    else:
        agent = KnowledgeGraphAgent(llm_provider="ollama")
        sample = (
            "Marie Curie discovered radium. Radium is used in cancer treatment. "
            "Albert Einstein developed the theory of relativity. "
            "Einstein was born in Ulm."
        )
        agent.ingest(sample)
        q = "What is used in cancer treatment?"
        print(agent.answer_query(q))
