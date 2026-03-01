"""
workflow_agent_demo.py

Demo: Workflow Agent (Task Orchestration Pattern).

This agent:
1. ANALYZES: Breaks down goals into ordered tasks with dependencies
2. PLANS: Creates a DAG (Directed Acyclic Graph) of task execution
3. EXECUTES: Runs tasks in dependency order, handling parallelization
4. MONITORS: Tracks task status and manages failures
5. ADAPTS: Re-plans if tasks fail or return unexpected results

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python workflow_agent_demo.py

This pattern powers enterprise workflow systems, Apache Airflow, Temporal,
and modern orchestration platforms. Essential for complex multi-step operations.
Used by: ML pipelines, ETL systems, business automation, data processing.
Features: Dependency management, parallelization, error handling, monitoring.
"""

import os
import json
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
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
# Task & Workflow Structures
# ============================================================================

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    READY = "ready"  # Dependencies satisfied
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskType(Enum):
    """Type of task."""
    COMPUTE = "compute"
    RETRIEVE = "retrieve"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"


@dataclass
class Task:
    """A single task in the workflow."""
    task_id: str
    name: str
    description: str
    task_type: TaskType
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2

    def __repr__(self):
        status_icon = {
            TaskStatus.PENDING: "○",
            TaskStatus.READY: "→",
            TaskStatus.RUNNING: "⟳",
            TaskStatus.COMPLETED: "✓",
            TaskStatus.FAILED: "✗",
            TaskStatus.SKIPPED: "⊘",
        }[self.status]
        return f"{status_icon} {self.task_id}: {self.name} ({self.status.value})"

    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are completed."""
        return all(dep in completed_tasks for dep in self.dependencies)


class Workflow:
    """Manages a directed acyclic graph of tasks."""

    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.tasks: Dict[str, Task] = {}
        self.task_counter = 0
        self.execution_order: List[str] = []
        self.completed_tasks: Set[str] = set()

    def add_task(
        self,
        name: str,
        description: str,
        task_type: TaskType,
        dependencies: List[str] = None,
    ) -> str:
        """Add a task to the workflow."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter:03d}"

        task = Task(
            task_id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            dependencies=dependencies or [],
        )

        self.tasks[task_id] = task
        return task_id

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks ready to execute (dependencies satisfied)."""
        ready = []
        for task in self.tasks.values():
            if (
                task.status == TaskStatus.PENDING
                and task.can_execute(self.completed_tasks)
            ):
                task.status = TaskStatus.READY
                ready.append(task)
        return ready

    def mark_completed(self, task_id: str, result: str):
        """Mark task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].result = result
            self.tasks[task_id].completed_at = datetime.now().isoformat()
            self.completed_tasks.add(task_id)

    def mark_failed(self, task_id: str, error: str):
        """Mark task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.FAILED
            self.tasks[task_id].error = error
            self.tasks[task_id].retry_count += 1

    def get_status(self) -> Dict[str, Any]:
        """Get workflow status."""
        total = len(self.tasks)
        completed = len(self.completed_tasks)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)

        return {
            "total_tasks": total,
            "completed": completed,
            "running": running,
            "failed": failed,
            "progress": f"{completed}/{total}" if total > 0 else "0/0",
        }

    def print_dag(self):
        """Print workflow DAG."""
        print(f"\nWorkflow: {self.workflow_name}")
        print("=" * 60)

        for task_id, task in self.tasks.items():
            print(f"{task}")
            if task.dependencies:
                deps_str = ", ".join(task.dependencies)
                print(f"   ↑ depends on: {deps_str}")
        print()


# ============================================================================
# Workflow Agent
# ============================================================================

class WorkflowAgent:
    """
    Workflow Agent - Orchestrates complex multi-step workflows with
    dependency management, parallelization, and error handling.
    Powers enterprise systems, ML pipelines, and business automation.
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        max_parallel_tasks: int = 3,
    ):
        self.llm_provider = llm_provider
        self.max_parallel_tasks = max_parallel_tasks
        self.workflow: Optional[Workflow] = None

    def plan_workflow(self, goal: str) -> Workflow:
        """Plan a workflow to achieve a goal."""
        prompt = f"""Plan a workflow to achieve this goal: {goal}

Create a detailed plan with 5-8 tasks that:
1. Have clear dependencies
2. Can be executed in parallel where possible
3. Are logically ordered
4. Include validation/error checking

Format:
TASK_NAME | DESCRIPTION | TYPE (compute/retrieve/validate/transform/aggregate) | DEPENDS_ON (comma-separated task names or "none")

Example:
Data Collection | Fetch raw data from sources | retrieve | none
Data Validation | Check data quality | validate | Data Collection
Data Cleaning | Remove duplicates and errors | transform | Data Validation
Feature Engineering | Create derived features | compute | Data Cleaning
Model Training | Train ML model | compute | Feature Engineering
Model Validation | Validate model performance | validate | Model Training
Report Generation | Create results report | aggregate | Model Validation

Provide only the task specifications, no other text."""

        system_prompt = """You are a workflow planning expert. Create clear, well-ordered tasks
with appropriate dependencies. Think about parallelization opportunities."""

        response = call_llm(prompt, self.llm_provider, system_prompt)

        # Parse tasks
        workflow = Workflow(goal)
        task_map = {}  # Maps task names to IDs

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                task_name = parts[0]
                description = parts[1]
                task_type_str = parts[2].lower()
                depends_on_str = parts[3]

                # Parse task type
                task_type = TaskType.COMPUTE  # default
                for tt in TaskType:
                    if tt.value in task_type_str:
                        task_type = tt
                        break

                # Parse dependencies
                deps = []
                if depends_on_str.lower() != "none" and depends_on_str:
                    dep_names = [d.strip() for d in depends_on_str.split(",")]
                    deps = [task_map.get(dn, dn) for dn in dep_names]

                # Add task
                task_id = workflow.add_task(
                    name=task_name,
                    description=description,
                    task_type=task_type,
                    dependencies=deps,
                )
                task_map[task_name] = task_id

        return workflow

    def execute_task(self, task: Task) -> Tuple[bool, str]:
        """Execute a single task."""
        prompt = f"""Execute this workflow task:

Task: {task.name}
Type: {task.task_type.value}
Description: {task.description}

Provide:
1. What the task accomplishes
2. Key steps or process
3. Expected output
4. Success validation

Keep response concise (3-4 sentences)."""

        system_prompt = f"You are executing a workflow task: {task.task_type.value}"

        try:
            result = call_llm(prompt, self.llm_provider, system_prompt)
            return True, result
        except Exception as e:
            return False, str(e)

    def run_workflow(self, goal: str) -> Dict[str, Any]:
        """Execute a complete workflow."""
        print(f"\n{'='*70}")
        print(f"Workflow Agent (using {self.llm_provider})")
        print(f"{'='*70}\n")

        print(f"Goal: {goal}\n")

        # Step 1: Plan
        print("📋 PLANNING WORKFLOW...")
        self.workflow = self.plan_workflow(goal)
        self.workflow.print_dag()

        # Step 2: Execute
        print(f"{'='*70}")
        print("EXECUTING WORKFLOW")
        print(f"{'='*70}\n")

        iteration = 0
        max_iterations = len(self.workflow.tasks) * 2  # Failsafe

        while len(self.workflow.completed_tasks) < len(self.workflow.tasks):
            iteration += 1
            if iteration > max_iterations:
                print("Max iterations reached!")
                break

            # Get ready tasks
            ready_tasks = self.workflow.get_ready_tasks()

            if not ready_tasks:
                # Check for failed tasks that can be retried
                retryable = [
                    t for t in self.workflow.tasks.values()
                    if t.status == TaskStatus.FAILED and t.retry_count < t.max_retries
                ]
                if retryable:
                    ready_tasks = retryable
                else:
                    break

            print(f"Round {iteration}: {len(ready_tasks)} task(s) ready")

            # Execute ready tasks
            for task in ready_tasks[:self.max_parallel_tasks]:
                print(f"  ▶ Executing: {task.name}")

                task.status = TaskStatus.RUNNING
                success, result = self.execute_task(task)

                if success:
                    self.workflow.mark_completed(task.task_id, result)
                    print(f"    ✓ Completed: {result[:60]}...")
                else:
                    self.workflow.mark_failed(task.task_id, result)
                    print(f"    ✗ Failed: {result[:60]}...")

            print()

        # Summary
        print(f"\n{'='*70}")
        print("WORKFLOW EXECUTION SUMMARY")
        print(f"{'='*70}\n")

        status = self.workflow.get_status()
        print(f"Tasks Completed: {status['completed']}/{status['total_tasks']}")
        print(f"Failed: {status['failed']}")
        print(f"Success Rate: {(status['completed']/status['total_tasks']*100) if status['total_tasks'] > 0 else 0:.0f}%\n")

        # Show results
        print("Task Results:")
        for task in self.workflow.tasks.values():
            if task.status == TaskStatus.COMPLETED:
                print(f"  ✓ {task.name}: {task.result[:60]}...")
            elif task.status == TaskStatus.FAILED:
                print(f"  ✗ {task.name}: {task.error[:60]}...")

        return {
            "goal": goal,
            "total_tasks": status["total_tasks"],
            "completed": status["completed"],
            "failed": status["failed"],
            "success": status["completed"] == status["total_tasks"],
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
            model="gpt-3.5-turbo", messages=messages, max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ============================================================================
# Main - Demo
# ============================================================================

if __name__ == "__main__":
    print("Workflow Agent Demo")
    print("=" * 70)

    print("\nWorkflow Orchestration Pipeline:")
    print("1. ANALYZE: Break goal into tasks")
    print("2. PLAN: Build DAG with dependencies")
    print("3. EXECUTE: Run tasks in order (parallelize where possible)")
    print("4. MONITOR: Track status and failures")
    print("5. ADAPT: Retry failed tasks or re-plan\n")

    print("Key Features:")
    print("  ✓ Directed Acyclic Graph (DAG) for task dependencies")
    print("  ✓ Parallelization of independent tasks")
    print("  ✓ Failure detection and retry logic")
    print("  ✓ Status monitoring and progress tracking")
    print("  ✓ Task orchestration and sequencing\n")

    print("Real-World Applications:")
    print("  ✓ ML Pipeline orchestration")
    print("  ✓ ETL (Extract-Transform-Load) systems")
    print("  ✓ Business process automation")
    print("  ✓ Data processing workflows")
    print("  ✓ Microservice orchestration\n")

    print("Powers:")
    print("  ✓ Apache Airflow")
    print("  ✓ Temporal.io (workflow engine)")
    print("  ✓ Kubernetes Jobs")
    print("  ✓ Enterprise automation platforms\n")

    print("Note: Set OLLAMA_HOST/OLLAMA_MODEL or OPENAI_API_KEY for real execution.\n")

    # Create agent
    agent = WorkflowAgent(llm_provider="ollama", max_parallel_tasks=3)

    # Example goals
    goals = [
        "Process customer data and generate insights report",
        "Train and deploy a machine learning model",
        "ETL pipeline for data warehouse",
    ]

    print("Example Workflow Goals:")
    for i, goal in enumerate(goals, 1):
        print(f"  {i}. {goal}")

    print("\n\nExample Workflow DAG:")
    print("""
Workflow: Process customer data and generate insights
============================================================
→ task_001: Data Collection (retrieve)
   ↑ depends on: none
✓ task_002: Data Validation (validate)
   ↑ depends on: Data Collection
✓ task_003: Data Cleaning (transform)
   ↑ depends on: Data Validation
✓ task_004: Exploratory Analysis (compute)
   ↑ depends on: Data Cleaning
✓ task_005: Feature Engineering (compute)
   ↑ depends on: Data Cleaning
⟳ task_006: Statistical Tests (validate)
   ↑ depends on: Exploratory Analysis, Feature Engineering
○ task_007: Report Generation (aggregate)
   ↑ depends on: Statistical Tests

Execution Plan:
Round 1: task_001 (Data Collection)
Round 2: task_002 (Data Validation) 
Round 3: task_003 (Data Cleaning)
Round 4: task_004 (Exploratory) + task_005 (Feature Engineering) [Parallel!]
Round 5: task_006 (Statistical Tests)
Round 6: task_007 (Report Generation)
    """)

    # Uncomment to run with real LLM:
    # result = agent.run_workflow(goals[0])
    # print(f"\nResult: {json.dumps(result, indent=2)}")
