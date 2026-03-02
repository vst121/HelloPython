"""
voice_ai_agent_demo.py

Demo: Voice AI Agent – handles voice-based interactions with speech-to-text
(STT), LLM processing, and text-to-speech (TTS) capabilities. Popularized
by voice assistants like Alexa, Siri, and conversational AI systems.

This demo shows how an agent might receive audio input, transcribe it to
text using a speech-to-text service, process it with an LLM, and synthesize
the response back to speech.

Flow:
1. Receive user voice input (audio file)
2. Transcribe audio to text using STT service
3. Process query with LLM
4. Synthesize response to audio using TTS service

In practice, STT/TTS might use services like OpenAI Whisper, Google Cloud
Speech, Amazon Polly, or local models like Coqui TTS. Here we mock them
to keep dependencies minimal.

Usage:
- pip install requests
- Set OLLAMA_HOST, OLLAMA_MODEL or OPENAI_API_KEY
- python voice_ai_agent_demo.py

"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

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
# Speech-to-Text component (mock)
# ---------------------------------------------------------------------------

def mock_speech_to_text(audio_path: str) -> str:
    """Simulate converting speech to text. In real use, this would call
    a speech recognition service (Whisper, Google Speech, etc.)."""
    # Very simplistic placeholder logic - in practice, use Whisper API
    base = os.path.basename(audio_path).lower()
    
    # Simulate different transcriptions based on filename hints
    if "hello" in base or "greeting" in base:
        return "Hello, how can you help me today?"
    elif "weather" in base:
        return "What's the weather like today?"
    elif "reminder" in base:
        return "Please remind me to buy groceries at 5pm."
    elif "music" in base:
        return "Play some relaxing jazz music."
    else:
        return "I'd like to know more about artificial intelligence."


# ---------------------------------------------------------------------------
# Text-to-Speech component (mock)
# ---------------------------------------------------------------------------

def mock_text_to_speech(text: str, output_path: str) -> str:
    """Simulate converting text to speech. In real use, this would call
    a TTS service (ElevenLabs, Amazon Polly, Coqui TTS, etc.)."""
    # Simulate audio file generation
    print(f"Synthesizing speech: '{text[:50]}...'")
    print(f"Audio would be saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Voice Agent implementation
# ---------------------------------------------------------------------------

@dataclass
class VoiceInteraction:
    """Represents a complete voice interaction."""
    audio_input: str
    transcribed_text: str
    llm_response: str
    audio_output: Optional[str] = None


class VoiceAIAgent:
    """Agent that handles end-to-end voice conversations."""
    
    def __init__(self, llm_provider: str = "ollama"):
        self.llm_provider = llm_provider
        self.conversation_history: list[VoiceInteraction] = []
        
        # Initialize LLM client based on provider
        if llm_provider == "openai" and OPENAI_AVAILABLE:
            self.client = OpenAI()
        elif OLLAMA_AVAILABLE:
            self.client = None  # Will use requests directly
        else:
            raise ImportError(
                "Either 'requests' (for Ollama) or 'openai' package required"
            )
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM to generate a response."""
        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise and conversational."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content
        
        # Default to Ollama
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise and conversational."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    def process_voice_input(
        self, 
        audio_path: str, 
        generate_speech: bool = True
    ) -> VoiceInteraction:
        """Process voice input and return the interaction result."""
        print(f"\n{'='*60}")
        print(f"Processing voice input: {audio_path}")
        print(f"{'='*60}")
        
        # Step 1: Speech-to-Text
        print("\n[1/3] Transcribing speech to text...")
        transcribed_text = mock_speech_to_text(audio_path)
        print(f"    Transcribed: {transcribed_text}")
        
        # Step 2: Process with LLM
        print("\n[2/3] Processing with LLM...")
        llm_response = self._call_llm(transcribed_text)
        print(f"    Response: {llm_response}")
        
        # Step 3: Text-to-Speech (optional)
        audio_output = None
        if generate_speech:
            print("\n[3/3] Synthesizing speech response...")
            output_filename = f"response_{len(self.conversation_history)}.wav"
            audio_output = mock_text_to_speech(llm_response, output_filename)
        
        # Store interaction
        interaction = VoiceInteraction(
            audio_input=audio_path,
            transcribed_text=transcribed_text,
            llm_response=llm_response,
            audio_output=audio_output
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
            summary += f"   Response: {interaction.llm_response}\n"
        
        return summary


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main():
    """Demonstrate the Voice AI Agent."""
    print("Voice AI Agent Demo")
    print("=" * 60)
    print("This demo simulates a voice AI agent that:")
    print("1. Receives voice input (audio file)")
    print("2. Transcribes speech to text (STT)")
    print("3. Processes with LLM")
    print("4. Synthesizes response to speech (TTS)")
    print("=" * 60)
    
    # Initialize agent
    agent = VoiceAIAgent(llm_provider="ollama")
    
    # Simulated voice inputs
    voice_inputs = [
        "greeting_hello.wav",
        "query_weather.wav",
        "task_reminder.wav",
        "preference_music.wav"
    ]
    
    # Process each voice input
    for audio_file in voice_inputs:
        try:
            result = agent.process_voice_input(audio_file, generate_speech=True)
            print(f"\n✓ Completed interaction")
        except Exception as e:
            print(f"\n✗ Error processing {audio_file}: {e}")
    
    # Show conversation summary
    print("\n" + "=" * 60)
    print("Conversation Summary")
    print("=" * 60)
    print(agent.get_conversation_summary())
    
    print("\n" + "=" * 60)
    print("Note: This demo uses mock STT/TTS. In production, integrate")
    print("real services like Whisper (STT) and ElevenLabs/Polly (TTS).")
    print("=" * 60)


if __name__ == "__main__":
    main()
