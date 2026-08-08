# Python Mastery

Welcome to **Python Mastery**, a comprehensive project designed to systematically review and master Python's capabilities.

## Project Mission

This project serves as a structured review of **Python abilities** and acts as a strategic **gate to AI Engineering**. It bridges the gap between core programming excellence and the specialized requirements of artificial intelligence and machine learning.

## Advanced Mastery Demonstrations

This project contains a series of specialized scripts exploring advanced Python concepts:

| File                                  | Topic                    | Description                                                                      |
| :------------------------------------ | :----------------------- | :------------------------------------------------------------------------------- |
| `args_kwargs_demo.py`                 | **Flexible Arguments**   | Use of `*args` and `**kwargs` for variable length inputs.                        |
| `dunder_methods_demo.py`              | **Magic Methods**        | Operator overloading and custom object behavior using `__str__`, `__add__`, etc. |
| `context_managers_demo.py`            | **Resource Management**  | Handling setup/teardown with `with` blocks and `@contextmanager`.                |
| `metaclasses_demo.py`                 | **Metaprogramming**      | Using metaclasses to control class creation and enforce standards.               |
| `abstract_base_classes_demo.py`       | **Interfaces (ABCs)**    | Defining strict blueprints and API contracts for subclasses.                     |
| `multiple_inheritance_demo.py`        | **MRO & Mixins**         | Understanding Method Resolution Order and composing behaviors with Mixins.       |
| `async_await_demo.py`                 | **Concurrency**          | Non-blocking execution and event loops using `asyncio`.                          |
| `monkey_patching_demo.py`             | **Runtime Modification** | Dynamically altering code behavior for testing or bug fixes.                     |
| `memory_management_demo.py`           | **Garbage Collection**   | Deep dive into reference counting and circular references.                       |
| `closures_demo.py`                    | **Closures**             | Function object remembering values in enclosing scopes for state and factories.  |
| `performance_optimization_demo.py`    | **Optimization**         | Performance techniques like `__slots__`, `lru_cache`, and local caching.         |
| `descriptors_demo.py`                 | **Descriptors**          | Deep dive into `__get__` and `__set__` for custom attribute access logic.        |
| `structural_pattern_matching_demo.py` | **Pattern Matching**     | Expert use of `match-case` for destructuring and validation (3.10+).             |
| `encapsulation_demo.py`               | **Encapsulation**        | Data protection with private attributes and properties.                          |
| `generators_demo.py`                  | **Generators**           | Memory-efficient iteration using `yield`.                                        |
| `type_hinting_demo.py`                | **Type Hinting**         | Static typing annotations for better code clarity and tooling.                   |
| `coroutines_demo.py`                  | **Coroutines**           | Advanced async patterns and producer-consumer patterns.                          |
| `decorators_demo.py`                  | **Decorators**           | Function and class decorators for cross-cutting concerns.                        |
| `docker_demo.py`                      | **Dockerization**        | Dockerizing a simple fastapi python project.                                     |
| `harnessing_demo.py`                  | **Test Harnessing**      | Building a reusable, dependency-free harness for running and timing cases.       |

## Core Patterns

- **Singleton Pattern**: `singleton_pattern_demo.py` - Ensuring single instances for loggers/configs.
- **Factory Pattern**: `factory_pattern_demo.py` - Object creation abstraction.
- **Strategy Pattern**: `strategy_pattern_demo.py` - Encapsulating interchangeable algorithms (e.g., discounts).
- **Command Pattern**: `command_pattern_demo.py` - Encapsulating requests as objects with undo support.
- **Chain of Responsibility**: `chain_of_responsibility_demo.py` - Passing requests along a chain of handlers.
- **Adapter Pattern**: `adapter_pattern_demo.py` - Bridging incompatible interfaces.
- **Proxy Pattern**: `proxy_pattern_demo.py` - Intercepting object access.
- **Composite Pattern**: `composite_pattern_demo.py` - Tree structures and part-whole hierarchies.

## Testing & Quality Assurance

- **Pytest**: `test_pytest_demo.py` - Demonstrating fixtures, parametrization, exception testing, and mocking for robust code verification.

## AI Agent Demos

This project includes extensive demonstrations of AI agent architectures using Large Language Models (LLMs):

### Foundation & Utility Agents

| File                       | Description                                          |
| :------------------------- | :--------------------------------------------------- |
| `ollama_chat_demo.py`      | Basic chat interaction with Ollama local LLM.        |
| `web_search_agent_demo.py` | Agent that searches the web for current information. |
| `sql_agent_demo.py`        | Natural language to SQL query generation.            |
| `ai_harnessing_demo.py`    | Offline evaluation harness for AI responses.         |

### Single-Agent Architectures

| File                                    | Description                                                         |
| :-------------------------------------- | :------------------------------------------------------------------ |
| `autonomous_agent_demo.py`              | Fully autonomous agent with tools and self-directed planning.       |
| `react_agent_demo.py`                   | ReAct (Reason + Act) pattern combining reasoning with tool actions. |
| `self_ask_agent_demo.py`                | Self-ask with search agent for complex question answering.          |
| `chain_of_verification_agent_demo.py`   | Multi-stage verification to ensure response accuracy.               |
| `function_calling_agent_demo.py`        | Using function calling to structure LLM outputs.                    |
| `recursive_decomposition_agent_demo.py` | Breaking complex problems into smaller subproblems.                 |

### Memory & Knowledge Agents

| File                             | Description                                                  |
| :------------------------------- | :----------------------------------------------------------- |
| `memory_augmented_agent_demo.py` | Agent with persistent memory for long-running conversations. |
| `rag_agent_demo.py`              | Retrieval-Augmented Generation for knowledge base queries.   |
| `knowledge_graph_agent_demo.py`  | Using knowledge graphs for structured information retrieval. |

### Multi-Agent Systems

| File                            | Description                                     |
| :------------------------------ | :---------------------------------------------- |
| `multi_agent_demo.py`           | Multiple agents collaborating on complex tasks. |
| `workflow_agent_demo.py`        | Orchestrating agents in defined workflows.      |
| `ensemble_voting_agent_demo.py` | Multiple agents voting/consensus for decisions. |

### Specialized Application Agents

| File                               | Description                                      |
| :--------------------------------- | :----------------------------------------------- |
| `ai_email_agent_demo.py`           | Composing and analyzing emails with AI.          |
| `data_analysis_agent_demo.py`      | Analyzing datasets and generating insights.      |
| `personal_assistant_agent_demo.py` | General-purpose personal assistant capabilities. |
| `prompt_engineering_agent_demo.py` | Assisting with prompt creation and optimization. |
| `scientific_agent_demo.py`         | Scientific research and literature analysis.     |
| `text_adventure_agent_demo.py`     | Interactive narrative and game experiences.      |

### Reasoning & Cognitive Agents

| File                                 | Description                                              |
| :----------------------------------- | :------------------------------------------------------- |
| `tree_of_thought_agent_demo.py`      | Exploring multiple reasoning paths simultaneously.       |
| `self_refining_agent_demo.py`        | Agent that iteratively improves its own outputs.         |
| `metacognitive_agent_demo.py`        | Agent with awareness of its own reasoning process.       |
| `socratic_questioning_agent_demo.py` | Using Socratic method for deeper understanding.          |
| `debate_agent_demo.py`               | Multiple agents debating topics for comprehensive views. |
| `scenario_planning_agent_demo.py`    | Exploring future scenarios and contingencies.            |

### Collaborative & Task-Specific Agents

| File                                | Description                                                            |
| :---------------------------------- | :--------------------------------------------------------------------- |
| `pair_programmer_agent_demo.py`     | AI pair programming assistant.                                         |
| `tdd_agent_demo.py`                 | Test-Driven Development with AI assistance.                            |
| `feedback_loop_agent_demo.py`       | Agent that learns from user feedback.                                  |
| `curriculum_agent_demo.py`          | Adaptive learning path creation.                                       |
| `persona_agent_demo.py`             | Maintaining consistent character/persona in conversations.             |
| `negotiation_agent_demo.py`         | Negotiation and conflict resolution.                                   |
| `counterfactual_agent_demo.py`      | Exploring "what-if" scenarios.                                         |
| `ethical_reasoning_agent_demo.py`   | Ethical analysis and reasoning.                                        |
| `context_manager_agent_demo.py`     | Managing context across complex interactions.                          |
| `multimodal_agent_demo.py`          | Processing multiple input types (text, images, etc.).                  |
| `voice_ai_agent_demo.py`            | Voice AI agent with function calling, streaming, JSON mode, and tools. |
| `voice_ai_agent_evaluation_demo.py` | Evaluation framework for Voice AI Agent with test cases and metrics.   |
| `agent_observability_demo.py`       | Observability framework with logging, metrics, tracing, and health.    |
| `ai_gateway_security_demo.py`       | AI Gateway with rate limiting, authentication, and threat detection.   |

## Getting Started

To begin your journey through Python mastery:

1. Clone the repository.
2. Run any demo file individually, e.g., `python async_await_demo.py`.
3. Explore the Jupyter notebooks in the root for interactive learning.

---

Targeted at developers transitioning into the world of AI Engineering.
