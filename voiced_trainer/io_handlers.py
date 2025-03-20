"""
Input/Output handlers for the VoicedTrainer application.

These handlers abstract the interaction with the user, allowing different
implementations (text, voice, etc.) to be used interchangeably.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI
from voiced_trainer.config import OPENAI_API_KEY, VOICE_INPUT_ENABLED, VOICE_OUTPUT_ENABLED

class InputHandler(ABC):
    """Abstract base class for input handlers."""
    
    @abstractmethod
    def get_input(self, prompt: str) -> str:
        """
        Get input from the user.
        
        Args:
            prompt: The prompt to show to the user
            
        Returns:
            The user's input as a string
        """
        pass


class OutputHandler(ABC):
    """Abstract base class for output handlers."""
    
    @abstractmethod
    def display_output(self, message: str) -> None:
        """
        Display output to the user.
        
        Args:
            message: The message to display
        """
        pass


class TextInputHandler(InputHandler):
    """Text-based input handler using the command line."""
    
    def get_input(self, prompt: str) -> str:
        """
        Get text input from the user via the command line.
        
        Args:
            prompt: The prompt to show to the user
            
        Returns:
            The user's input as a string
        """
        return input(f"{prompt} ").strip()


class TextOutputHandler(OutputHandler):
    """Text-based output handler using the command line."""
    
    def display_output(self, message: str) -> None:
        """
        Display text output on the command line.
        
        Args:
            message: The message to display
        """
        print(message)


class VoiceInputHandler(InputHandler):
    """Voice-based input handler using OpenAI's speech-to-text API."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """
        Initialize the voice input handler.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        
    def get_input(self, prompt: str) -> str:
        """
        Get voice input from the user using OpenAI's speech-to-text.
        
        Args:
            prompt: The prompt to show to the user
            
        Returns:
            The transcribed user's input as a string
        """
        print(f"{prompt} (Speak now...)")
        
        # Use a temporary file to store the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Import here to avoid dependencies if voice is not used
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            
            # Record audio (3 seconds by default)
            fs = 44100  # Sample rate
            seconds = 10  # Duration
            
            print("Recording...")
            recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            sd.wait()  # Wait until recording is finished
            print("Recording finished")
            
            # Save as WAV file
            sf.write(temp_filename, recording, fs)
            
            # Transcribe with OpenAI
            with open(temp_filename, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            transcribed_text = transcript.text
            print(f"Transcribed: {transcribed_text}")
            return transcribed_text
        
        except Exception as e:
            print(f"Error during voice input: {e}")
            print("Falling back to text input...")
            return input("Type your response: ").strip()
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


class VoiceOutputHandler(OutputHandler):
    """Voice-based output handler using OpenAI's text-to-speech API."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """
        Initialize the voice output handler.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        
    def display_output(self, message: str) -> None:
        """
        Display output as synthesized speech using OpenAI's text-to-speech.
        
        Args:
            message: The message to convert to speech
        """
        print(message)  # Always print the message as well
        
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Generate speech
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=message
            )
            
            # Save to file
            response.stream_to_file(temp_filename)
            
            # Play the audio
            self._play_audio(temp_filename)
            
        except Exception as e:
            print(f"Error during voice output: {e}")
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    def _play_audio(self, file_path: str) -> None:
        """
        Play an audio file.
        
        Args:
            file_path: Path to the audio file
        """
        try:
            # Import here to avoid dependencies if voice is not used
            from pydub import AudioSegment
            from pydub.playback import play
            
            sound = AudioSegment.from_mp3(file_path)
            play(sound)
            
        except Exception as e:
            print(f"Error playing audio: {e}")


def get_input_handler() -> InputHandler:
    """
    Factory function to get the appropriate input handler based on configuration.
    
    Returns:
        An instance of InputHandler
    """
    if VOICE_INPUT_ENABLED:
        return VoiceInputHandler()
    else:
        return TextInputHandler()


def get_output_handler() -> OutputHandler:
    """
    Factory function to get the appropriate output handler based on configuration.
    
    Returns:
        An instance of OutputHandler
    """
    if VOICE_OUTPUT_ENABLED:
        return VoiceOutputHandler()
    else:
        return TextOutputHandler()
