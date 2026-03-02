"""
voice_ai_agent_evaluation_demo.py

Evaluation Demo: Voice AI Agent Evaluation Framework

This demo provides a comprehensive evaluation framework for testing and measuring
the performance of the Voice AI Agent across various dimensions.

This demo showcases:
- Test Case Framework: Structured test scenarios with expected outcomes
- Performance Metrics: Response time, accuracy, tool usage statistics
- Quality Assessment: Response quality, error handling, edge cases
- Evaluation Report: Comprehensive results with visualizations

Flow:
1. Define test cases with expected outcomes
2. Run each test case against the Voice AI Agent
3. Collect metrics (accuracy, latency, tool usage)
4. Generate evaluation report with findings

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python voice_ai_agent_evaluation_demo.py

"""

import os
import json
import time
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

# Import from the main voice AI agent demo
from voice_ai_agent_demo import (
    VoiceAIAgent,
    VoiceInteraction,
    TOOL_DEFINITIONS,
    mock_speech_to_text,
    execute_tool
)


# ---------------------------------------------------------------------------
# Evaluation Metrics and Results
# ---------------------------------------------------------------------------

class TestStatus(Enum):
    """Status of a test case execution."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """A single test case for evaluating the Voice AI Agent."""
    name: str
    audio_input: str
    expected_intent: str
    expected_tools: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class TestResult:
    """Result of a single test case execution."""
    test_name: str
    status: TestStatus
    execution_time_ms: float
    transcribed_text: str = ""
    llm_response: str = ""
    tools_used: list[str] = field(default_factory=list)
    expected_tools_found: list[str] = field(default_factory=list)
    expected_tools_missing: list[str] = field(default_factory=list)
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    intent_matched: bool = False
    error_message: str = ""
    tool_execution_success: bool = True
    response_quality_score: float = 0.0


@dataclass
class EvaluationMetrics:
    """Aggregate metrics across all test cases."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    average_response_time_ms: float = 0.0
    tool_usage_accuracy: float = 0.0
    keyword_detection_accuracy: float = 0.0
    intent_matching_accuracy: float = 0.0
    overall_quality_score: float = 0.0
    tool_execution_success_rate: float = 0.0


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def get_evaluation_test_cases() -> list[TestCase]:
    """Define comprehensive test cases for Voice AI Agent evaluation."""
    
    return [
        TestCase(
            name="Weather Query",
            audio_input="audio_weather_query.wav",
            expected_intent="get_weather",
            expected_tools=["get_weather"],
            expected_keywords=["weather", "temperature", "forecast"],
            description="Test weather information retrieval"
        ),
        TestCase(
            name="Reminder Setting",
            audio_input="audio_reminder.wav",
            expected_intent="set_reminder",
            expected_tools=["set_reminder"],
            expected_keywords=["remind", "reminder", "remember"],
            description="Test reminder creation functionality"
        ),
        TestCase(
            name="Music Playback",
            audio_input="audio_music.wav",
            expected_intent="play_music",
            expected_tools=["play_music"],
            expected_keywords=["play", "music", "song"],
            description="Test music playback request"
        ),
        TestCase(
            name="Timer Setting",
            audio_input="audio_timer.wav",
            expected_intent="set_timer",
            expected_tools=["set_timer"],
            expected_keywords=["timer", "countdown", "minutes"],
            description="Test timer creation"
        ),
        TestCase(
            name="General Question",
            audio_input="audio_general.wav",
            expected_intent="answer_question",
            expected_tools=[],
            expected_keywords=["information", "help", "know"],
            description="Test general question without tool usage"
        ),
        TestCase(
            name="Complex Multi-Tool",
            audio_input="audio_complex.wav",
            expected_intent="multi_tool",
            expected_tools=["get_weather", "set_reminder"],
            expected_keywords=["weather", "remind", "later"],
            description="Test complex request requiring multiple tools"
        ),
        TestCase(
            name="Greeting",
            audio_input="audio_hello.wav",
            expected_intent="greeting",
            expected_tools=[],
            expected_keywords=["hello", "hi", "help"],
            description="Test simple greeting without tools"
        ),
        TestCase(
            name="Ambiguous Request",
            audio_input="audio_ambiguous.wav",
            expected_intent="ambiguous",
            expected_tools=[],
            expected_keywords=[],
            description="Test handling of ambiguous input"
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------

class VoiceAIEvaluationEngine:
    """Comprehensive evaluation engine for Voice AI Agent."""
    
    def __init__(self, agent: VoiceAIAgent):
        self.agent = agent
        self.test_results: list[TestResult] = []
        self.metrics = EvaluationMetrics()
    
    def evaluate_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case and collect metrics."""
        
        print(f"\n{'='*60}")
        print(f"Running Test: {test_case.name}")
        print(f"Description: {test_case.description}")
        print(f"{'='*60}")
        
        result = TestResult(
            test_name=test_case.name,
            status=TestStatus.SKIPPED,
            execution_time_ms=0.0
        )
        
        start_time = time.time()
        
        try:
            # Execute the voice interaction
            interaction = self.agent.process_voice_input(
                audio_path=test_case.audio_input,
                generate_speech=False,
                use_tools=True
            )
            
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            # Record transcribed text and response
            result.transcribed_text = interaction.transcribed_text
            result.llm_response = interaction.llm_response
            
            # Extract tools used from tool_calls
            for tc in interaction.tool_calls:
                if isinstance(tc, dict):
                    func_name = tc.get("function", {}).get("name") if "function" in tc else tc.get("name", "")
                else:
                    func_name = tc.function.name
                if func_name:
                    result.tools_used.append(func_name)
            
            # Check expected tools
            result.expected_tools_found = [
                tool for tool in test_case.expected_tools 
                if tool in result.tools_used
            ]
            result.expected_tools_missing = [
                tool for tool in test_case.expected_tools 
                if tool not in result.tools_used
            ]
            
            # Check keywords in response
            response_lower = interaction.llm_response.lower()
            result.keywords_found = [
                keyword for keyword in test_case.expected_keywords
                if keyword.lower() in response_lower
            ]
            result.keywords_missing = [
                keyword for keyword in test_case.expected_keywords
                if keyword.lower() not in response_lower
            ]
            
            # Check intent matching (based on tools used)
            result.intent_matched = (
                len(result.expected_tools_found) == len(test_case.expected_tools)
                if test_case.expected_tools else
                len(result.keywords_found) > 0 if test_case.expected_keywords else
                True  # No specific expectations
            )
            
            # Check tool execution success
            result.tool_execution_success = (
                len(interaction.tool_results) > 0 if result.tools_used else True
            )
            
            # Calculate response quality score (0-100)
            quality_score = 50.0  # Base score
            
            # Intent matching (+20)
            if result.intent_matched:
                quality_score += 20
            
            # Tools used correctly (+15)
            if len(result.expected_tools_missing) == 0:
                quality_score += 15
            
            # Keywords present (+10)
            if test_case.expected_keywords:
                keyword_ratio = len(result.keywords_found) / len(test_case.expected_keywords)
                quality_score += keyword_ratio * 10
            
            # Response length appropriate (+5)
            if 20 < len(interaction.llm_response) < 500:
                quality_score += 5
            
            result.response_quality_score = quality_score
            
            # Determine status
            if result.intent_matched and len(result.expected_tools_missing) == 0:
                result.status = TestStatus.PASSED
            elif result.intent_matched:
                result.status = TestStatus.PASSED  # Partial success
            else:
                result.status = TestStatus.FAILED
            
            print(f"\n✅ Test completed in {execution_time:.2f}ms")
            print(f"   Intent Matched: {result.intent_matched}")
            print(f"   Tools Found: {result.tools_used}")
            print(f"   Quality Score: {result.response_quality_score:.1f}/100")
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            result.execution_time_ms = (time.time() - start_time) * 1000
            print(f"\n❌ Test error: {e}")
        
        return result
    
    def run_evaluation(self, test_cases: list[TestCase]) -> EvaluationMetrics:
        """Run all test cases and compute aggregate metrics."""
        
        print("\n" + "="*60)
        print("VOICE AI AGENT EVALUATION")
        print("="*60)
        print(f"Total test cases: {len(test_cases)}")
        print(f"Agent Provider: {self.agent.llm_provider}")
        
        self.test_results = []
        
        # Run each test case
        for test_case in test_cases:
            result = self.evaluate_test_case(test_case)
            self.test_results.append(result)
        
        # Compute aggregate metrics
        self._compute_metrics()
        
        return self.metrics
    
    def _compute_metrics(self):
        """Calculate aggregate metrics from test results."""
        
        self.metrics.total_tests = len(self.test_results)
        
        # Count statuses
        for result in self.test_results:
            if result.status == TestStatus.PASSED:
                self.metrics.passed += 1
            elif result.status == TestStatus.FAILED:
                self.metrics.failed += 1
            elif result.status == TestStatus.ERROR:
                self.metrics.errors += 1
            else:
                self.metrics.skipped += 1
        
        # Calculate average response time
        valid_times = [r.execution_time_ms for r in self.test_results if r.status != TestStatus.ERROR]
        self.metrics.average_response_time_ms = (
            sum(valid_times) / len(valid_times) if valid_times else 0
        )
        
        # Calculate tool usage accuracy
        tool_tests = [r for r in self.test_results if r.expected_tools]
        if tool_tests:
            tool_correct = sum(1 for r in tool_tests if len(r.expected_tools_missing) == 0)
            self.metrics.tool_usage_accuracy = (tool_correct / len(tool_tests)) * 100
        
        # Calculate keyword detection accuracy
        keyword_tests = [r for r in self.test_results if r.expected_keywords]
        if keyword_tests:
            keyword_correct = sum(1 for r in keyword_tests if len(r.keywords_missing) == 0)
            self.metrics.keyword_detection_accuracy = (keyword_correct / len(keyword_tests)) * 100
        
        # Calculate intent matching accuracy
        intent_tests = [r for r in self.test_results if r.expected_intent]
        if intent_tests:
            intent_correct = sum(1 for r in intent_tests if r.intent_matched)
            self.metrics.intent_matching_accuracy = (intent_correct / len(intent_tests)) * 100
        
        # Calculate overall quality score
        quality_scores = [r.response_quality_score for r in self.test_results if r.status != TestStatus.ERROR]
        self.metrics.overall_quality_score = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0
        )
        
        # Calculate tool execution success rate
        tool_exec_tests = [r for r in self.test_results if r.tools_used]
        if tool_exec_tests:
            success_count = sum(1 for r in tool_exec_tests if r.tool_execution_success)
            self.metrics.tool_execution_success_rate = (success_count / len(tool_exec_tests)) * 100
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        
        report_lines = [
            "\n" + "="*60,
            "VOICE AI AGENT EVALUATION REPORT",
            "="*60,
            f"\n📊 EXECUTION SUMMARY",
            f"   Total Tests: {self.metrics.total_tests}",
            f"   ✅ Passed: {self.metrics.passed}",
            f"   ❌ Failed: {self.metrics.failed}",
            f"   ⚠️  Errors: {self.metrics.errors}",
            f"   ⏭️  Skipped: {self.metrics.skipped}",
            f"   Pass Rate: {(self.metrics.passed / self.metrics.total_tests * 100):.1f}%",
            
            f"\n⏱️  PERFORMANCE METRICS",
            f"   Avg Response Time: {self.metrics.average_response_time_ms:.2f}ms",
            
            f"\n🎯 ACCURACY METRICS",
            f"   Intent Matching: {self.metrics.intent_matching_accuracy:.1f}%",
            f"   Tool Usage: {self.metrics.tool_usage_accuracy:.1f}%",
            f"   Keyword Detection: {self.metrics.keyword_detection_accuracy:.1f}%",
            f"   Tool Execution Success: {self.metrics.tool_execution_success_rate:.1f}%",
            
            f"\n⭐ QUALITY SCORE",
            f"   Overall Quality: {self.metrics.overall_quality_score:.1f}/100",
            
            f"\n📋 DETAILED RESULTS",
        ]
        
        for result in self.test_results:
            status_icon = {
                TestStatus.PASSED: "✅",
                TestStatus.FAILED: "❌",
                TestStatus.ERROR: "⚠️",
                TestStatus.SKIPPED: "⏭️"
            }[result.status]
            
            report_lines.append(
                f"\n   {status_icon} {result.test_name}: {result.status.value}"
            )
            
            if result.transcribed_text:
                report_lines.append(f"      Input: {result.transcribed_text[:60]}...")
            
            if result.tools_used:
                report_lines.append(f"      Tools Used: {', '.join(result.tools_used)}")
            
            if result.expected_tools_missing:
                report_lines.append(f"      Missing Tools: {', '.join(result.expected_tools_missing)}")
            
            report_lines.append(f"      Quality: {result.response_quality_score:.1f}/100")
        
        report_lines.append("\n" + "="*60)
        
        return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Additional Evaluation: Stress Testing
# ---------------------------------------------------------------------------

@dataclass
class StressTestResult:
    """Results from stress testing the Voice AI Agent."""
    concurrent_requests: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    throughput: float  # requests per second
    timeout_count: int = 0


def run_stress_test(
    agent: VoiceAIAgent,
    num_requests: int = 10,
    concurrent: bool = False
) -> StressTestResult:
    """Run stress test on the Voice AI Agent."""
    
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {num_requests} requests")
    print(f"{'='*60}")
    
    test_audio = "audio_weather_query.wav"
    latencies = []
    successes = 0
    failures = 0
    timeouts = 0
    
    for i in range(num_requests):
        try:
            start = time.time()
            agent.process_voice_input(
                audio_path=test_audio,
                generate_speech=False,
                use_tools=True
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            successes += 1
            print(f"   Request {i+1}/{num_requests}: {latency:.2f}ms ✅")
        except Exception as e:
            failures += 1
            if "timeout" in str(e).lower():
                timeouts += 1
            print(f"   Request {i+1}/{num_requests}: FAILED ❌")
    
    total_time = sum(latencies)
    throughput = num_requests / total_time if total_time > 0 else 0
    
    return StressTestResult(
        concurrent_requests=1,
        total_requests=num_requests,
        successful_requests=successes,
        failed_requests=failures,
        average_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        throughput=throughput * 1000,  # Convert to requests per second
        timeout_count=timeouts
    )


# ---------------------------------------------------------------------------
# Main Evaluation Execution
# ---------------------------------------------------------------------------

def run_voice_ai_evaluation():
    """Run comprehensive evaluation of the Voice AI Agent."""
    
    print("\n" + "="*60)
    print("🎙️ VOICE AI AGENT EVALUATION SUITE")
    print("="*60)
    
    # Initialize the Voice AI Agent
    agent = VoiceAIAgent(
        llm_provider="ollama",
        temperature=0.7,
        top_p=0.9,
        enable_streaming=False,
        enable_json_mode=False
    )
    
    # Get test cases
    test_cases = get_evaluation_test_cases()
    
    # Run evaluation
    engine = VoiceAIEvaluationEngine(agent)
    metrics = engine.run_evaluation(test_cases)
    
    # Generate and print report
    report = engine.generate_report()
    print(report)
    
    # Run stress test
    stress_result = run_stress_test(agent, num_requests=5)
    print(f"\n📈 STRESS TEST RESULTS")
    print(f"   Total Requests: {stress_result.total_requests}")
    print(f"   Successful: {stress_result.successful_requests}")
    print(f"   Failed: {stress_result.failed_requests}")
    print(f"   Avg Latency: {stress_result.average_latency_ms:.2f}ms")
    print(f"   Throughput: {stress_result.throughput:.2f} req/s")
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 EVALUATION COMPLETE")
    print("="*60)
    print(f"\nOverall Assessment:")
    if metrics.overall_quality_score >= 80:
        print("   🏆 EXCELLENT - Agent performs at a high level")
    elif metrics.overall_quality_score >= 60:
        print("   ✅ GOOD - Agent meets most requirements")
    elif metrics.overall_quality_score >= 40:
        print("   ⚠️  NEEDS IMPROVEMENT - Several issues detected")
    else:
        print("   ❌ POOR - Significant improvements required")
    
    return metrics


if __name__ == "__main__":
    run_voice_ai_evaluation()
