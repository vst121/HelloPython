"""
autonomous_agent_demo.py

Demo: Autonomous AI Agent (AutoGPT/BabyAGI style).

This agent:
1. ANALYZES: Breaks down goals into sub-tasks
2. PLANS: Creates an execution plan with priorities
3. EXECUTES: Runs tasks with persistence (memory & context)
4. REFLECTS: Learns from outcomes and adjusts strategy
5. LOOPS: Continues until goal is achieved or max iterations reached

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python autonomous_agent_demo.py

This pattern powers AutoGPT, BabyAGI, and other popular autonomous agents.
Features: Long-running autonomy, memory, reflection, self-correction.
"""

import os
import json
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

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
# Memory and Knowledge Storage
# ============================================================================

class AgentMemory:
    """Simple memory system for the agent."""

    def __init__(self):
        self.short_term = []  # Recent context
        self.long_term = {}  # Facts and learnings
        self.task_history = []  # Completed tasks
        self.errors = []  # Errors encountered

    def add_short_term(self, content: str, context: str = ""):
        """Add to working memory."""
        self.short_term.append(
            {
                "timestamp": datetime.now().isoformat(),
                "content": content,
                "context": context,
            }
        )
        # Keep only last 10 items
        if len(self.short_term) > 10:
            self.short_term.pop(0)

    def add_long_term(self, key: str, value: str):
        """Store persistent knowledge."""
        self.long_term[key] = {"value": value, "learned_at": datetime.now().isoformat()}

    def add_task_result(self, task: str, result: str, success: bool):
        """Log completed task."""
        self.task_history.append(
            {
                "task": task,
                "result": result,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def add_error(self, error: str):
        """Log errors for reflection."""
        self.errors.append(
            {"error": error, "timestamp": datetime.now().isoformat()}
        )

    def get_context(self) -> str:
        """Get current context summary for prompts."""
        context = "=== Agent Memory ===\n"
        context += f"Tasks completed: {len(self.task_history)}\n"
        context += f"Errors: {len(self.errors)}\n"
        if self.long_term:
            context += "Knowledge: " + ", ".join(self.long_term.keys()) + "\n"
        if self.short_term:
            context += "Recent: " + self.short_term[-1].get("content", "")[:100] + "\n"
        return context


# ============================================================================
# Task Management
# ============================================================================

class Task:
    """Represents a task to be executed."""

    def __init__(self, task_id: int, name: str, description: str, priority: int = 1):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.priority = priority  # 1=highest, 5=lowest
        self.status = "pending"  # pending, in_progress, completed, failed
        self.result = None
        self.created_at = datetime.now().isoformat()
        self.completed_at = None

    def to_dict(self) -> Dict:
        return {
            "id": self.task_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
        }

    def __repr__(self):
        return f"Task({self.task_id}, {self.name}, priority={self.priority}, status={self.status})"


class TaskQueue:
    """Manages task queue with prioritization."""

    def __init__(self):
        self.tasks = []
        self.next_id = 1

    def add_task(self, name: str, description: str, priority: int = 1) -> Task:
        """Add a new task."""
        task = Task(self.next_id, name, description, priority)
        self.tasks.append(task)
        self.next_id += 1
        return task

    def get_next_task(self) -> Optional[Task]:
        """Get highest priority pending task."""
        pending = [t for t in self.tasks if t.status == "pending"]
        if pending:
            return min(pending, key=lambda x: x.priority)
        return None

    def mark_completed(self, task_id: int, result: str):
        """Mark task as completed."""
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = "completed"
                task.result = result
                task.completed_at = datetime.now().isoformat()
                return True
        return False

    def mark_failed(self, task_id: int, error: str):
        """Mark task as failed."""
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = "failed"
                task.result = error
                task.completed_at = datetime.now().isoformat()
                return True
        return False


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
            model="gpt-3.5-turbo", messages=messages, max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Autonomous Agent
# ============================================================================

class AutonomousAgent:
    """
    Autonomous Agent that can break down goals, plan, execute, and reflect.
    Similar to AutoGPT / BabyAGI but simplified for demo.
    """

    def __init__(self, llm_provider: str = "ollama", max_iterations: int = 5):
        self.llm_provider = llm_provider
        self.max_iterations = max_iterations
        self.memory = AgentMemory()
        self.task_queue = TaskQueue()
        self.iteration = 0

    def analyze_goal(self, goal: str) -> List[str]:
        """Break down the goal into sub-tasks."""
        prompt = f"""You are an autonomous AI agent. Your goal is: {goal}

Break this goal into 3-5 actionable sub-tasks. Return ONLY a numbered list.
Example:
1. Task 1 description
2. Task 2 description
3. Task 3 description"""

        system_prompt = "You are an expert planner. Be concise and practical."
        response = call_llm(prompt, self.llm_provider, system_prompt)
        
        # Parse tasks from response
        tasks = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number and dot
                task = re.sub(r'^\d+\.\s*', '', line)
                if task:
                    tasks.append(task)
        
        return tasks if tasks else [goal]

    def plan_execution(self, tasks: List[str]) -> List[Task]:
        """Create execution plan with priorities."""
        for i, task_desc in enumerate(tasks):
            priority = i + 1  # First task is highest priority
            self.task_queue.add_task(
                name=f"Subtask {i+1}",
                description=task_desc,
                priority=priority
            )
        
        self.memory.add_short_term(f"Created {len(tasks)} subtasks")
        return self.task_queue.tasks

    def execute_task(self, task: Task) -> str:
        """Execute a single task."""
        task.status = "in_progress"
        
        prompt = f"""Task: {task.description}

Context: {self.memory.get_context()}

How should this task be accomplished? Provide a clear, practical approach or result.
Be concise (2-3 sentences max)."""

        system_prompt = "You are an autonomous AI agent executing tasks. Be practical and efficient."
        result = call_llm(prompt, self.llm_provider, system_prompt)
        
        if result.startswith("(Error") or result.startswith("(Ollama"):
            self.task_queue.mark_failed(task.task_id, result)
            self.memory.add_error(f"Task {task.task_id} failed: {result}")
            return result
        
        self.task_queue.mark_completed(task.task_id, result)
        self.memory.add_task_result(task.name, result, success=True)
        return result

    def reflect(self) -> str:
        """Reflect on progress and adjust strategy."""
        completed = len([t for t in self.task_queue.tasks if t.status == "completed"])
        failed = len([t for t in self.task_queue.tasks if t.status == "failed"])
        
        prompt = f"""Review progress so far:
- Completed: {completed} tasks
- Failed: {failed} tasks
- Errors: {len(self.memory.errors)}

Recent results: {json.dumps(self.memory.task_history[-2:], indent=2) if self.memory.task_history else 'None yet'}

What adjustments should be made to succeed? Be brief."""

        system_prompt = "You are an autonomous agent reflecting on progress. Be strategic."
        reflection = call_llm(prompt, self.llm_provider, system_prompt)
        
        self.memory.add_short_term(f"Reflection: {reflection}", "reflection")
        return reflection

    def run(self, goal: str) -> Dict[str, Any]:
        """Main agent loop: Analyze → Plan → Execute → Reflect → Repeat."""
        print(f"\n{'='*70}")
        print(f"Autonomous Agent (using {self.llm_provider})")
        print(f"{'='*70}")
        print(f"\nGoal: {goal}\n")

        # Step 1: Analyze
        print("📊 ANALYZING GOAL...")
        tasks = self.analyze_goal(goal)
        print(f"Identified {len(tasks)} sub-tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task[:60]}")

        # Step 2: Plan
        print("\n📋 CREATING EXECUTION PLAN...")
        self.plan_execution(tasks)
        print(f"Task queue ready with {len(self.task_queue.tasks)} items")

        # Step 3-5: Execute & Reflect Loop
        for self.iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*50}")
            print(f"Iteration {self.iteration}/{self.max_iterations}")
            print(f"{'='*50}")

            # Get next task
            task = self.task_queue.get_next_task()
            if not task:
                print("✓ All tasks completed or failed!")
                break

            print(f"\n▶ Executing: {task.name}")
            print(f"   Description: {task.description[:70]}")

            result = self.execute_task(task)
            print(f"   Result: {result[:100]}")

            # Reflect periodically
            if self.iteration % 2 == 0 or self.iteration == 1:
                print("\n🤔 REFLECTING...")
                reflection = self.reflect()
                print(f"   Insight: {reflection[:100]}")

        # Summary
        print(f"\n{'='*70}")
        print("EXECUTION SUMMARY")
        print(f"{'='*70}")
        
        completed = len([t for t in self.task_queue.tasks if t.status == "completed"])
        failed = len([t for t in self.task_queue.tasks if t.status == "failed"])
        
        print(f"Total tasks: {len(self.task_queue.tasks)}")
        print(f"Completed: {completed} ✓")
        print(f"Failed: {failed} ✗")
        print(f"Success rate: {completed}/{len(self.task_queue.tasks)}")

        return {
            "goal": goal,
            "iterations": self.iteration,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "memory_size": len(self.memory.long_term),
        }


# ============================================================================
# Main - Demo
# ============================================================================

if __name__ == "__main__":
    print("Autonomous Agent Demo")
    print("="*70)
    
    # Example goal
    goal = "Create a plan to learn Python programming from scratch and build a simple project"
    
    # Create agent
    agent = AutonomousAgent(llm_provider="ollama", max_iterations=3)
    
    # Run agent
    print("\nNote: This demo shows the agent pattern.")
    print("Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real LLM execution.\n")
    
    # For demo, show the expected flow:
    print("Expected Agent Flow:")
    print("1. ANALYZE: Break goal into subtasks")
    print("2. PLAN: Create prioritized task queue")
    print("3. EXECUTE: Run tasks sequentially")
    print("4. REFLECT: Analyze progress and adjust")
    print("5. LOOP: Continue until goal achieved\n")
    
    # Uncomment to run with real LLM:
    # result = agent.run(goal)
    # print(f"\nAgent Result: {result}")
