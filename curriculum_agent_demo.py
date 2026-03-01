"""
curriculum_agent_demo.py

Demo: Curriculum Agent – an adaptive tutoring agent that builds a
learning curriculum tailored to the user’s current knowledge and
difficulty preference. This pattern is widely used in educational
platforms, language learning apps, and intelligent tutoring systems.

Flow:
1. User provides topic and self-assessed skill level (beginner,
   intermediate, advanced)
2. Agent generates a sequence of lessons or modules using the LLM
3. After each lesson, agent quizzes the user and assesses performance
4. Difficulty adjusts based on quiz results; the curriculum evolves

This approach is popular because it personalizes learning and keeps
students engaged by matching their pace.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python curriculum_agent_demo.py

"""

import os
from typing import List, Dict

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


class CurriculumAgent:
    """Builds and adapts a curriculum based on user feedback."""

    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.lessons: List[Dict] = []
        self.current_index: int = 0

    def generate_curriculum(self, topic: str, level: str) -> None:
        prompt = (
            "You are an educational designer. Create a sequence of 5 lessons "
            "for a user learning about a topic. Each lesson should include a "
            "brief description and one quiz question. Use the user's skill "
            f"level: {level}.\n\nTopic: {topic}\n\nCurriculum:"
        )
        response = call_llm(prompt, self.llm_provider)
        # naive parsing: split by lines with "Lesson" label
        lessons = []
        for block in response.split("Lesson")[1:]:
            text = block.strip()
            if not text:
                continue
            lines = text.split("\n")
            desc = lines[0].strip(".: ")
            quiz = "".join(lines[1:]).strip()
            lessons.append({"description": desc, "quiz": quiz})
        self.lessons = lessons
        self.current_index = 0

    def present_next_lesson(self) -> Dict:
        if self.current_index < len(self.lessons):
            lesson = self.lessons[self.current_index]
            self.current_index += 1
            return lesson
        return {}

    def adjust_difficulty(self, correct: bool) -> None:
        if correct and self.current_index < len(self.lessons):
            # if user answered correctly, skip ahead or add bonus
            self.current_index += 1
        elif not correct and self.current_index > 1:
            # revisit previous lesson
            self.current_index = max(0, self.current_index - 1)

    def has_more(self) -> bool:
        return self.current_index < len(self.lessons)


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
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1500)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = CurriculumAgent(llm_provider="ollama")
    topic = "basic Python programming"
    level = "beginner"
    agent.generate_curriculum(topic, level)

    print("Generated curriculum:")
    for i, lesson in enumerate(agent.lessons, 1):
        print(f"Lesson {i}: {lesson['description']}")

    # simulate going through curriculum with user responding
    while agent.has_more():
        lesson = agent.present_next_lesson()
        print(f"\nLesson: {lesson['description']}")
        print(f"Quiz: {lesson['quiz']}")
        # mock user always correct for demo
        agent.adjust_difficulty(correct=True)

    print("\nCurriculum complete!")
