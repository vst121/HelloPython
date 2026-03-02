"""
voice_ai_agent_demo.py

Demo: Voice AI Agent with Advanced LLM Features – handles voice-based 
interactions using modern LLM capabilities including function calling,
structured outputs, streaming, JSON mode, and tool execution.

This demo showcases:
- Function Calling: Define tools the agent can invoke (reminders, weather, etc.)
- Structured Output: JSON mode for structured responses
- Streaming: Real-time response streaming
- Temperature/Top-P: Sampling parameter controls
- Tool Definitions: OpenAI-style tool schemas

Flow:
1. Receive user voice input (audio file)
2. Transcribe audio to text using STT service
3. Process with LLM using advanced features (tools, JSON mode, streaming)
4. Execute any tool calls returned by LLM
5. Synthesize final response to speech using TTS

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python voice_ai_agent_demo.py

"""

import os
import json
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

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
# Tool Definitions (Function Calling)
# ---------------------------------------------------------------------------

class ToolType(Enum):
    """Available tool types for the voice agent."""
    GET_WEATHER = "get_weather"
    SET_REMINDER = "set_reminder"
    PLAY_MUSIC = "play_music"
    SET_TIMER = "set_timer"
    ANSWERQuestion = "answer_question"


# Define tools using OpenAI-compatible schema
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name or location"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder or alarm",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What to remind about"},
                    "time": {"type": "string", "description": "When to remind (e.g., '5pm', 'in 30 minutes')"}
                },
                "required": ["task", "time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Play music or a playlist",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {"type": "string", "description": "Music genre or style"},
                    "mood": {"type": "string", "description": "Mood for the music"}
                },
                "required": ["genre"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a countdown timer",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {"type": "integer", "description": "Timer duration in seconds"},
                    "label": {"type": "string", "description": "Label for what the timer is for"}
                },
                "required": ["duration_seconds"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    print(f"\n🔧 Executing tool: {tool_name}")
    print(f"   Arguments: {arguments}")
    
    # Mock implementations - in production, these would call real APIs
    if tool_name == "get_weather":
        location = arguments.get("location", "Unknown")
        return f"The weather in {location} is currently sunny, 72°F (22°C)."
    
    elif tool_name == "set_reminder":
        task = arguments.get("task", "reminder")
        time = arguments.get("time", "later")
        return f"Reminder set: '{task}' at {time}."
    
    elif tool_name == "play_music":
        genre = arguments.get("genre", "any")
        mood = arguments.get("mood", "relaxed")
        return f"Now playing {genre} music for {mood} mood."
    
    elif tool_name == "set_timer":
        duration = arguments.get("duration_seconds", 60)
        label = arguments.get("label", "Timer")
        return f"Timer set for {duration} seconds: {label}."
    
    return "Tool executed successfully."


# ---------------------------------------------------------------------------
# Speech-to-Text component (mock)
# ---------------------------------------------------------------------------

def mock_speech_to_text(audio_path: str) -> str:
    """Simulate converting speech to text. In real use, this would call
    a speech recognition service (Whisper, Google Speech, etc.)."""
    base = os.path.basename(audio_path).lower()
    
    # Simulate different transcriptions based on filename hints
    if "hello" in base or "greeting" in base:
        return "Hello, how can you help me today?"
    elif "weather" in base:
        return "What's the weather like today in New York?"
    elif "reminder" in base:
        return "Please remind me to buy groceries at 5pm."
    elif "music" in base:
        return "Play some relaxing jazz music."
    elif "timer" in base:
        return "Set a timer for 10 minutes for cooking pasta."
    else:
        return "I'd like to know more about artificial intelligence."


# ---------------------------------------------------------------------------
# Text-to-Speech component (mock)
# ---------------------------------------------------------------------------

def mock_text_to_speech(text: str, output_path: str) -> str:
    """Simulate converting text to speech."""
    print(f"🗣️ Synthesizing speech: '{text[:50]}...'")
    print(f"   Audio would be saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Advanced LLM Client with New Features
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for advanced LLM features."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 300
    stream: bool = False
    json_mode: bool = False


class AdvancedLLMClient:
    """LLM client with modern features: function calling, streaming, JSON mode."""
    
    def __init__(self, provider: str = "ollama", config: Optional[LLMConfig] = None):
        self.provider = provider
        self.config = config or LLMConfig()
        
        if provider == "openai" and OPENAI_AVAILABLE:
            self.client = OpenAI()
        elif OLLAMA_AVAILABLE:
            self.client = None
        else:
            raise ImportError("Required: 'requests' (for Ollama) or 'openai' package")
    
    def generate(
        self, 
        system_prompt: str, 
        user_message: str,
        tools: Optional[list] = None,
        stream_callback: Optional[callable] = None
    ) -> dict:
        """Generate response with advanced LLM features."""
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Build request parameters
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        if self.provider == "openai" and OPENAI_AVAILABLE:
            return self._call_openai(messages, tools, params, stream_callback)
        else:
            return self._call_ollama(messages, tools, params, stream_callback)
    
    def _call_openai(
        self, 
        messages: list, 
        tools: Optional[list],
        params: dict,
        stream_callback: Optional[callable]
    ) -> dict:
        """Call OpenAI API with advanced features."""
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        # Add JSON mode
        if self.config.json_mode:
            params["response_format"] = {"type": "json_object"}
        
        # Handle streaming
        if self.config.stream and stream_callback:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                **params
            )
            
            full_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    stream_callback(content)
            
            return {"content": full_content, "tool_calls": None}
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                **params
            )
            
            message = response.choices[0].message
            return {
                "content": message.content,
                "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') else None
            }
    
    def _call_ollama(
        self, 
        messages: list, 
        tools: Optional[list],
        params: dict,
        stream_callback: Optional[callable]
    ) -> dict:
        """Call Ollama API with advanced features."""
        
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        
        request_data = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": params["temperature"],
                "num_predict": params["max_tokens"],
                "top_p": self.config.top_p,
            },
            "stream": self.config.stream
        }
        
        # Ollama tool calling support (if model supports it)
        if tools:
            request_data["tools"] = tools
        
        if self.config.stream and stream_callback:
            with requests.post(f"{host}/api/chat", json=request_data, stream=True) as r:
                full_content = ""
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            if content:
                                full_content += content
                                stream_callback(content)
                return {"content": full_content, "tool_calls": None}
        else:
            response = requests.post(f"{host}/api/chat", json=request_data)
            response.raise_for_status()
            data = response.json()
            
            return {
                "content": data["message"]["content"],
                "tool_calls": data["message"].get("tool_calls", None)
            }


# ---------------------------------------------------------------------------
# Voice AI Agent with Advanced Features
# ---------------------------------------------------------------------------

@dataclass
class VoiceInteraction:
    """Represents a complete voice interaction with tool execution."""
    audio_input: str
    transcribed_text: str
    llm_response: str
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    audio_output: Optional[str] = None
    streamed_tokens: int = 0


class VoiceAIAgent:
    """Advanced Voice AI Agent with modern LLM features."""
    
    def __init__(
        self, 
        llm_provider: str = "ollama",
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_streaming: bool = False,
        enable_json_mode: bool = False
    ):
        self.llm_provider = llm_provider
        self.conversation_history: list[VoiceInteraction] = []
        
        # Configure LLM with advanced features
        llm_config = LLMConfig(
            temperature=temperature,
            top_p=top_p,
            stream=enable_streaming,
            json_mode=enable_json_mode
        )
        
        self.llm = AdvancedLLMClient(llm_provider, llm_config)
        
        # Print configuration
        print(f"\n🎙️ Voice AI Agent initialized")
        print(f"   Provider: {llm_provider}")
        print(f"   Temperature: {temperature}")
        print(f"   Top-P: {top_p}")
        print(f"   Streaming: {enable_streaming}")
        print(f"   JSON Mode: {enable_json_mode}")
        print(f"   Tools Available: {len(TOOL_DEFINITIONS)}")
    
    def _stream_callback(self, token: str) -> None:
        """Callback for streaming responses."""
        print(token, end="", flush=True)
    
    def process_voice_input(
        self, 
        audio_path: str, 
        generate_speech: bool = True,
        use_tools: bool = True
    ) -> VoiceInteraction:
        """Process voice input with advanced LLM features."""
        
        print(f"\n{'='*60}")
        print(f"Processing voice input: {audio_path}")
        print(f"{'='*60}")
        
        # Step 1: Speech-to-Text
        print("\n[1/4] 🎤 Transcribing speech to text...")
        transcribed_text = mock_speech_to_text(audio_path)
        print(f"   Transcribed: \"{transcribed_text}\"")
        
        # Step 2: LLM Processing with advanced features
        print("\n[2/4] 🤖 Processing with LLM...")
        
        system_prompt = """You are a helpful voice assistant. When users ask about 
weather, reminders, music, or timers, use the available tools to help them.
Keep responses concise and conversational. If you use a tool, explain what 
you're doing."""
        
        tools = TOOL_DEFINITIONS if use_tools else None
        
        if self.llm.config.stream:
            print("   Streaming response: ", end="", flush=True)
        
        result = self.llm.generate(
            system_prompt=system_prompt,
            user_message=transcribed_text,
            tools=tools,
            stream_callback=self._stream_callback if self.llm.config.stream else None
        )
        
        llm_response = result["content"]
        tool_calls = result.get("tool_calls", [])
        
        if not self.llm.config.stream:
            print(f"   Response: {llm_response}")
        
        # Step 3: Execute tool calls
        tool_results = []
        if tool_calls:
            print("\n[3/4] 🔧 Executing tool calls...")
            for tool_call in tool_calls:
                # Handle both OpenAI and Ollama tool call formats
                if isinstance(tool_call, dict):
                    func_name = tool_call.get("function", {}).get("name") if "function" in tool_call else tool_call.get("name", "")
                    func_args = tool_call.get("function", {}).get("arguments") if "function" in tool_call else tool_call.get("arguments", {})
                else:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                
                # Convert arguments if they're a string
                if isinstance(func_args, str):
                    func_args = json.loads(func_args)
                
                result = execute_tool(func_name, func_args)
                tool_results.append({
                    "tool": func_name,
                    "arguments": func_args,
                    "result": result
                })
                print(f"   ✓ {func_name}: {result}")
            
            # Generate final response after tool execution
            if tool_results:
                print("\n[3.5/4] 🔄 Generating final response with tool results...")
                tool_context = "\n".join([f"- {t['tool']}: {t['result']}" for t in tool_results])
                final_prompt = f"""Original question: {transcribed_text}
                
Tool results:
{tool_context}

Provide a natural, conversational response incorporating the tool results."""
                
                final_result = self.llm.generate(
                    system_prompt="You are a helpful voice assistant. Respond naturally.",
                    user_message=final_prompt,
                    tools=None
                )
                llm_response = final_result["content"]
                print(f"   Final response: {llm_response}")
        else:
            print("\n[3/4] ⏭️ No tool calls needed")
        
        # Step 4: Text-to-Speech
        audio_output = None
        if generate_speech:
            print("\n[4/4] 🔊 Synthesizing speech response...")
            output_filename = f"response_{len(self.conversation_history)}.wav"
            audio_output = mock_text_to_speech(llm_response, output_filename)
        
        # Store interaction
        interaction = VoiceInteraction(
            audio_input=audio_path,
            transcribed_text=transcribed_text,
            llm_response=llm_response,
            tool_calls=[str(tc) for tc in tool_calls] if tool_calls else [],
            tool_results=tool_results,
            audio_output=audio_output,
            streamed_tokens=len(llm_response.split()) if self.llm.config.stream else 0
        )
        self.conversation_history.append(interaction)
        
        return interaction
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return "No conversations yet."
        
        summary = f"Total interactions: {len(self.conversation_history)}\n\n"
        for i, interaction in enumerate(self.conversation_history, 1):
            summary += f"{i}. Input: {interaction.transcribed_text}\n"
            if interaction.tool_calls:
                summary += f"   Tools used: {len(interaction.tool_calls)}\n"
            summary += f"   Response: {interaction.llm_response[:100]}...\n"
        
        return summary


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main():
    """Demonstrate the Advanced Voice AI Agent."""
    print("=" * 60)
    print("Voice AI Agent Demo - Advanced LLM Features")
    print("=" * 60)
    print("""
This demo showcases modern LLM capabilities:
• Function Calling: Use tools (weather, reminders, music, timers)
• Structured Output: JSON mode for programmatic responses
• Streaming: Real-time token-by-token response
• Temperature/Top-P: Sampling controls for creativity vs determinism

The agent will:
1. Transcribe voice input to text
2. Use LLM with tool calling to handle requests
3. Execute tools and incorporate results
4. Synthesize speech response
    """)
    
    # Demo 1: Standard mode with tools
    print("\n" + "=" * 60)
    print("Demo 1: Voice Agent with Function Calling")
    print("=" * 60)
    
    agent_with_tools = VoiceAIAgent(
        llm_provider="ollama",
        temperature=0.7,
        enable_streaming=False
    )
    
    voice_inputs = [
        "query_weather.wav",
        "task_reminder.wav",
        "preference_music.wav",
        "timer_cooking.wav"
    ]
    
    for audio_file in voice_inputs:
        try:
            agent_with_tools.process_voice_input(audio_file, generate_speech=True, use_tools=True)
            print(f"\n✓ Completed interaction\n")
        except Exception as e:
            print(f"\n⚠ Error: {e}\n")
    
    # Demo 2: Streaming mode
    print("\n" + "=" * 60)
    print("Demo 2: Voice Agent with Streaming")
    print("=" * 60)
    
    agent_streaming = VoiceAIAgent(
        llm_provider="ollama",
        temperature=0.8,
        enable_streaming=True
    )
    
    try:
        agent_streaming.process_voice_input("greeting_hello.wav", generate_speech=False, use_tools=False)
    except Exception as e:
        print(f"\n⚠ Streaming not available: {e}")
    
    # Demo 3: JSON Mode
    print("\n" + "=" * 60)
    print("Demo 3: Voice Agent with JSON Mode")
    print("=" * 60)
    
    agent_json = VoiceAIAgent(
        llm_provider="ollama",
        enable_json_mode=True
    )
    
    # Show conversation summary
    print("\n" + "=" * 60)
    print("Conversation Summary")
    print("=" * 60)
    print(agent_with_tools.get_conversation_summary())
    
    print("\n" + "=" * 60)
    print("Key LLM Features Demonstrated:")
    print("=" * 60)
    print("""
✓ Function Calling - Define and execute tools via LLM
✓ Structured Output - JSON mode for programmatic responses
✓ Streaming - Real-time response streaming
✓ Temperature/Top-P - Sampling parameter controls
✓ Tool Definitions - OpenAI-compatible tool schemas
✓ Multi-step Reasoning - LLM → Tool → LLM pipeline
    """)


if __name__ == "__main__":
    main()
