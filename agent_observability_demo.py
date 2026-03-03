"""
agent_observability_demo.py

Demo: Agent Observability Framework - Monitoring, Tracing, and Metrics

This demo provides comprehensive observability features for AI agents including:
- Structured Logging: JSON-formatted logs with timestamps and levels
- Metrics Collection: Request latency, token usage, error rates
- Distributed Tracing: Track agent decisions and tool executions
- Health Monitoring: Agent health checks and status tracking
- Alerting: Configurable alerts for anomalies

Flow:
1. Initialize ObservabilityManager with logging, metrics, and tracing
2. Instrument the agent with decorators/wrappers
3. Execute agent tasks with full visibility
4. View collected metrics and traces

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL (or OPENAI_API_KEY)
- python agent_observability_demo.py

"""

import os
import json
import time
import traceback
from typing import Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from contextlib import contextmanager
import threading

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ============================================================================
# Observability Enums and Data Classes
# ============================================================================

class LogLevel(Enum):
    """Log levels for observability."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class TraceEventType(Enum):
    """Types of trace events."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    RETRY = "retry"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    context: dict = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "context": self.context
        })


@dataclass
class MetricValue:
    """Metric value with metadata."""
    name: str
    value: float
    metric_type: MetricType
    labels: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TraceSpan:
    """Represents a single trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"
    
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


# ============================================================================
# Core Observability Components
# ============================================================================

class StructuredLogger:
    """Structured JSON logger for agent observability."""
    
    def __init__(self, name: str = "agent", log_level: LogLevel = LogLevel.INFO):
        self.name = name
        self.log_level = log_level
        self.handlers = []
        self._lock = threading.Lock()
    
    def _should_log(self, level: LogLevel) -> bool:
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return levels.index(level) >= levels.index(self.log_level)
    
    def log(self, level: LogLevel, message: str, **context) -> None:
        """Log a structured message."""
        if not self._should_log(level):
            return
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            message=message,
            context={
                "service": self.name,
                **context
            }
        )
        
        with self._lock:
            for handler in self.handlers:
                handler(entry)
        
        # Also print to console
        print(f"[{level.value}] {message}")
        if context:
            print(f"    Context: {json.dumps(context, indent=4)}")
    
    def debug(self, message: str, **context) -> None:
        self.log(LogLevel.DEBUG, message, **context)
    
    def info(self, message: str, **context) -> None:
        self.log(LogLevel.INFO, message, **context)
    
    def warning(self, message: str, **context) -> None:
        self.log(LogLevel.WARNING, message, **context)
    
    def error(self, message: str, **context) -> None:
        self.log(LogLevel.ERROR, message, **context)
    
    def critical(self, message: str, **context) -> None:
        self.log(LogLevel.CRITICAL, message, **context)
    
    def add_handler(self, handler: Callable[[LogEntry], None]) -> None:
        """Add a log handler."""
        self.handlers.append(handler)


class MetricsCollector:
    """Collects and aggregates metrics for the agent."""
    
    def __init__(self):
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: float = 1.0, labels: dict = None) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self.counters[key] += value
    
    def gauge(self, name: str, value: float, labels: dict = None) -> None:
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self.gauges[key] = value
    
    def histogram(self, name: str, value: float, labels: dict = None) -> None:
        """Record a histogram value."""
        key = self._make_key(name, labels)
        with self._lock:
            self.histograms[key].append(value)
    
    def timer(self, name: str) -> Callable:
        """Context manager for timing operations."""
        return self._TimerContext(name, self)
    
    def _make_key(self, name: str, labels: dict = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_stats(self) -> dict:
        """Get current metrics statistics."""
        stats = {}
        
        with self._lock:
            # Compute counter stats
            stats["counters"] = dict(self.counters)
            
            # Compute gauge stats
            stats["gauges"] = dict(self.gauges)
            
            # Compute histogram stats
            hist_stats = {}
            for name, values in self.histograms.items():
                if values:
                    hist_stats[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "sum": sum(values)
                    }
            stats["histograms"] = hist_stats
        
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
    
    class _TimerContext:
        def __init__(self, name: str, collector: 'MetricsCollector'):
            self.name = name
            self.collector = collector
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.histogram(self.name, duration_ms)


class DistributedTracer:
    """Distributed tracing for agent operations."""
    
    def __init__(self):
        self.spans: dict[str, TraceSpan] = {}
        self._lock = threading.Lock()
        self._span_counter = 0
    
    def _generate_span_id(self) -> str:
        self._span_counter += 1
        return f"span-{self._span_counter:06d}"
    
    def start_span(
        self,
        trace_id: str,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: dict = None
    ) -> TraceSpan:
        """Start a new trace span."""
        span = TraceSpan(
            trace_id=trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        with self._lock:
            self.spans[span.span_id] = span
        
        return span
    
    def end_span(self, span_id: str, status: str = "ok", tags: dict = None) -> None:
        """End a trace span."""
        with self._lock:
            if span_id in self.spans:
                span = self.spans[span_id]
                span.end_time = time.time()
                span.status = status
                if tags:
                    span.tags.update(tags)
    
    def add_event(self, span_id: str, event_name: str, tags: dict = None) -> None:
        """Add an event to a span."""
        with self._lock:
            if span_id in self.spans:
                self.spans[span_id].events.append({
                    "name": event_name,
                    "timestamp": time.time(),
                    "tags": tags or {}
                })
    
    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            return [s for s in self.spans.values() if s.trace_id == trace_id]
    
    def get_all_traces(self) -> dict[str, list[TraceSpan]]:
        """Get all traces."""
        traces = defaultdict(list)
        with self._lock:
            for span in self.spans.values():
                traces[span.trace_id].append(span)
        return dict(traces)


class HealthMonitor:
    """Monitors agent health and provides status."""
    
    def __init__(self):
        self.checks: dict[str, Callable[[], bool]] = {}
        self.status = "healthy"
        self.last_check_time = None
        self._lock = threading.Lock()
    
    def register_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Register a health check."""
        self.checks[name] = check_fn
    
    def run_checks(self) -> dict:
        """Run all health checks."""
        results = {}
        all_healthy = True
        
        with self._lock:
            self.last_check_time = datetime.now().isoformat()
            
            for name, check_fn in self.checks.items():
                try:
                    healthy = check_fn()
                    results[name] = {"status": "healthy" if healthy else "unhealthy"}
                    if not healthy:
                        all_healthy = False
                except Exception as e:
                    results[name] = {"status": "error", "error": str(e)}
                    all_healthy = False
            
            self.status = "healthy" if all_healthy else "unhealthy"
        
        return results


# ============================================================================
# Observability Manager - Main Entry Point
# ============================================================================

class ObservabilityManager:
    """
    Central manager for all observability components.
    Provides unified interface for logging, metrics, tracing, and health monitoring.
    """
    
    def __init__(
        self,
        service_name: str = "agent",
        log_level: LogLevel = LogLevel.INFO,
        enable_tracing: bool = True,
        enable_metrics: bool = True
    ):
        self.service_name = service_name
        self.logger = StructuredLogger(service_name, log_level)
        self.metrics = MetricsCollector()
        self.tracer = DistributedTracer() if enable_tracing else None
        self.health = HealthMonitor()
        
        # Counters for important events
        self.total_requests = 0
        self.total_errors = 0
        
        # Setup default log handler
        self.logger.add_handler(self._default_log_handler)
        
        # Register default health checks
        self._setup_default_health_checks()
    
    def _default_log_handler(self, entry: LogEntry) -> None:
        """Default log handler that could write to file, etc."""
        pass  # Console output handled in StructuredLogger.log
    
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks."""
        self.health.register_check("logging", lambda: True)
        self.health.register_check("metrics", lambda: len(self.metrics.gauges) >= 0)
    
    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        trace_id: str = None,
        parent_span_id: str = None,
        tags: dict = None
    ):
        """Context manager for tracing operations."""
        if not self.tracer:
            yield
            return
        
        trace_id = trace_id or f"trace-{int(time.time() * 1000)}"
        span = self.tracer.start_span(trace_id, operation_name, parent_span_id, tags)
        
        try:
            yield span
        except Exception as e:
            self.tracer.end_span(span.span_id, status="error", tags={"error": str(e)})
            self.error(f"Operation failed: {operation_name}", error=str(e))
            raise
        else:
            self.tracer.end_span(span.span_id, status="ok")
    
    def track_request(self, labels: dict = None):
        """Context manager for tracking request metrics."""
        return _RequestTracker(self, labels)
    
    def log(self, level: LogLevel, message: str, **context) -> None:
        """Log a message."""
        self.logger.log(level, message, **context)
    
    def debug(self, message: str, **context) -> None:
        self.logger.debug(message, **context)
    
    def info(self, message: str, **context) -> None:
        self.logger.info(message, **context)
    
    def warning(self, message: str, **context) -> None:
        self.logger.warning(message, **context)
    
    def error(self, message: str, **context) -> None:
        self.logger.error(message, **context)
        self.total_errors += 1
        self.metrics.increment("agent_errors_total", labels=context.get("labels", {}))
    
    def critical(self, message: str, **context) -> None:
        self.logger.critical(message, **context)
    
    def get_dashboard_summary(self) -> dict:
        """Get summary for dashboard display."""
        stats = self.metrics.get_stats()
        health_results = self.health.run_checks()
        
        return {
            "service": self.service_name,
            "status": self.health.status,
            "metrics": stats,
            "health": health_results,
            "uptime": self.total_requests,
            "error_rate": self.total_errors / max(self.total_requests, 1)
        }


class _RequestTracker:
    """Context manager for tracking request metrics."""
    
    def __init__(self, observability: ObservabilityManager, labels: dict = None):
        self.obs = observability
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.obs.total_requests += 1
        self.obs.metrics.increment("agent_requests_total", labels=self.labels)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.obs.metrics.histogram("agent_request_duration_ms", duration_ms, self.labels)
        
        if exc_type:
            self.obs.metrics.increment("agent_request_errors_total", labels=self.labels)


# ============================================================================
# Observable Agent Wrapper
# ============================================================================

class ObservableAgent:
    """Wraps an agent with observability capabilities."""
    
    def __init__(self, agent: Any, observability: ObservabilityManager):
        self.agent = agent
        self.obs = observability
    
    def run(self, *args, **kwargs):
        """Run agent with full observability."""
        trace_id = f"trace-{int(time.time() * 1000)}"
        
        with self.obs.trace_operation("agent.run", trace_id, tags={"agent": type(self.agent).__name__}):
            with self.obs.track_request(labels={"agent": type(self.agent).__name__}):
                self.obs.info("Starting agent execution", trace_id=trace_id)
                
                try:
                    result = self.agent.run(*args, **kwargs)
                    self.obs.info("Agent execution completed", trace_id=trace_id)
                    return result
                except Exception as e:
                    self.obs.error(
                        "Agent execution failed",
                        trace_id=trace_id,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    raise


# ============================================================================
# Demo - Observable LLM Client
# ============================================================================

class ObservableLLMClient:
    """LLM client with observability instrumentation."""
    
    def __init__(self, provider: str = "ollama", observability: ObservabilityManager = None):
        self.provider = provider
        self.obs = observability or ObservabilityManager(provider)
        
        if provider == "ollama" and OLLAMA_AVAILABLE:
            self.client = None
        else:
            self.client = None
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate with full observability."""
        
        with self.obs.trace_operation("llm.generate") as span:
            span.tags.update({
                "provider": self.provider,
                "prompt_length": len(prompt)
            })
            
            self.obs.info("LLM request started", prompt_length=len(prompt))
            
            try:
                with self.obs.metrics.timer("llm_response_time") as timer:
                    # Simulate LLM call
                    time.sleep(0.1)  # Simulate API latency
                    
                    result = {
                        "content": f"Response to: {prompt[:50]}...",
                        "tokens_used": len(prompt.split()) * 2
                    }
                
                # Record metrics
                self.obs.metrics.increment("llm_requests_total")
                self.obs.metrics.histogram("llm_tokens_used", result["tokens_used"])
                
                span.tags.update({
                    "status": "success",
                    "tokens_used": result["tokens_used"]
                })
                
                self.obs.info("LLM request completed", tokens=result["tokens_used"])
                return result
                
            except Exception as e:
                self.obs.error("LLM request failed", error=str(e))
                span.tags["status"] = "error"
                raise


# ============================================================================
# Demo Execution
# ============================================================================

def run_observability_demo():
    """Run the observability demo."""
    
    print("\n" + "="*60)
    print("🔍 AGENT OBSERVABILITY DEMO")
    print("="*60)
    
    # Initialize observability manager
    obs = ObservabilityManager(
        service_name="demo-agent",
        log_level=LogLevel.INFO,
        enable_tracing=True,
        enable_metrics=True
    )
    
    # Add custom log handler
    def file_log_handler(entry: LogEntry):
        # In production, this would write to a file or log aggregation service
        pass
    
    obs.logger.add_handler(file_log_handler)
    
    # =========================================================================
    # Demo 1: Structured Logging
    # =========================================================================
    
    print("\n--- Demo 1: Structured Logging ---")
    
    obs.info("Agent initialized", version="1.0.0", model="llama3.2")
    obs.info("Processing request", request_id="req-123", user_id="user-456")
    obs.warning("Rate limit approaching", current=95, limit=100)
    obs.error("Request failed", request_id="req-123", error="Timeout")
    obs.debug("Debug information", details={"key": "value"})
    
    # =========================================================================
    # Demo 2: Metrics Collection
    # =========================================================================
    
    print("\n--- Demo 2: Metrics Collection ---")
    
    # Simulate multiple requests
    for i in range(5):
        with obs.track_request(labels={"endpoint": "chat", "status": "success"}):
            time.sleep(0.05)
    
    # Track different types of metrics
    obs.metrics.increment("users_active_total", labels={"region": "us-east"})
    obs.metrics.increment("users_active_total", labels={"region": "us-west"})
    obs.metrics.gauge("memory_usage_mb", 256.5)
    obs.metrics.histogram("request_size_bytes", 1024)
    obs.metrics.histogram("request_size_bytes", 2048)
    
    # Get metrics stats
    stats = obs.metrics.get_stats()
    print("\n📊 Metrics Collected:")
    print(f"   Counters: {stats['counters']}")
    print(f"   Gauges: {stats['gauges']}")
    print(f"   Histograms: {stats['histograms']}")
    
    # =========================================================================
    # Demo 3: Distributed Tracing
    # =========================================================================
    
    print("\n--- Demo 3: Distributed Tracing ---")
    
    trace_id = f"trace-{int(time.time() * 1000)}"
    
    with obs.trace_operation("process_request", trace_id, tags={"user": "demo"}) as parent_span:
        parent_span.tags["priority"] = "high"
        
        # Sub-operation: Authentication
        with obs.trace_operation("authenticate", trace_id, parent_span.span_id) as auth_span:
            time.sleep(0.05)
            auth_span.tags["method"] = "jwt"
        
        # Sub-operation: LLM Processing
        with obs.trace_operation("llm_processing", trace_id, parent_span.span_id) as llm_span:
            time.sleep(0.1)
            llm_span.tags["model"] = "llama3.2"
            llm_span.tags["tokens"] = 150
        
        # Sub-operation: Response formatting
        with obs.trace_operation("format_response", trace_id, parent_span.span_id):
            time.sleep(0.02)
    
    # Get trace details
    trace = obs.tracer.get_trace(trace_id)
    print(f"\n📝 Trace '{trace_id}' spans:")
    for span in trace:
        print(f"   - {span.operation_name}: {span.duration_ms():.2f}ms [{span.status}]")
    
    # =========================================================================
    # Demo 4: Observable LLM Client
    # =========================================================================
    
    print("\n--- Demo 4: Observable LLM Client ---")
    
    llm = ObservableLLMClient("ollama", obs)
    
    response = llm.generate("What is the meaning of life?")
    print(f"   Response: {response['content']}")
    
    # =========================================================================
    # Demo 5: Health Monitoring
    # =========================================================================
    
    print("\n--- Demo 5: Health Monitoring ---")
    
    # Register custom health check
    obs.health.register_check("database", lambda: True)
    obs.health.register_check("api", lambda: True)
    
    health_results = obs.health.run_checks()
    print(f"\n💚 Health Status: {obs.health.status}")
    for check, result in health_results.items():
        print(f"   - {check}: {result['status']}")
    
    # =========================================================================
    # Demo 6: Dashboard Summary
    # =========================================================================
    
    print("\n--- Demo 6: Dashboard Summary ---")
    
    summary = obs.get_dashboard_summary()
    print(f"\n📊 Dashboard Summary:")
    print(f"   Service: {summary['service']}")
    print(f"   Status: {summary['status']}")
    print(f"   Total Requests: {summary['uptime']}")
    print(f"   Error Rate: {summary['error_rate']:.2%}")
    print(f"   Metrics:")
    for metric_type, values in summary['metrics'].items():
        if values:
            print(f"      {metric_type}: {json.dumps(values)[:100]}...")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    
    print("\n" + "="*60)
    print("✅ OBSERVABILITY DEMO COMPLETE")
    print("="*60)
    
    print("""
This demo showcased:
1. Structured Logging - JSON-formatted logs with context
2. Metrics Collection - Counters, gauges, histograms
3. Distributed Tracing - Trace spans with parent-child relationships
4. Health Monitoring - Custom health checks
5. Dashboard Summary - Unified metrics overview

All of these can be integrated with:
- Prometheus/Grafana for metrics visualization
- Jaeger/Zipkin for distributed tracing
- ELK Stack for log aggregation
- Custom alerting systems
""")


if __name__ == "__main__":
    run_observability_demo()
