"""
Input/Output handlers for the VoicedTrainer application.

These handlers abstract the interaction with the user, allowing different
implementations (text, voice, etc.) to be used interchangeably.
"""

import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict

from openai import OpenAI
from voiced_trainer.config import (
    OPENAI_API_KEY, 
    VOICE_INPUT_ENABLED, 
    VOICE_OUTPUT_ENABLED,
    TEMP_DIR,
    LANGUAGE
)

# Deutsch/Englisch Übersetzungen für allgemeine Texte
TRANSLATIONS = {
    "de": {
        "speak_now": "Sprechen Sie jetzt...",
        "recording": "Aufnahme... (Sprechen Sie oder drücken Sie Strg+C zum Beenden)",
        "auto_stopped": "Aufnahme automatisch gestoppt nach Stille.",
        "manual_stopped": "Aufnahme manuell gestoppt.",
        "recording_saved": "Aufnahme beendet und in {} gespeichert.",
        "no_recording": "Keine Aufnahme gemacht.",
        "enter_answer": "Bitte geben Sie Ihre Antwort ein:",
        "transcribing": "Transkribiere Audio...",
        "transcription": "Transkription: \"{}\"",
        "confirm_transcription": "Ist diese Transkription korrekt? (J/N, oder editieren Sie den Text):",
        "rejected": "Transkription abgelehnt. Bitte geben Sie Ihre Antwort ein:",
        "recording_error": "Fehler während der Sprachaufnahme: {error}",
        "fallback": "Fallback auf Texteingabe...",
        "temp_file_error": "Warnung: Konnte temporäre Datei nicht löschen: {error}",
        "generating_speech": "Sprachausgabe wird generiert...",
        "playing_audio": "Audio wird abgespielt...",
        "audio_error": "Fehler bei der Sprachausgabe: {error}",
        "playback_complete": "Audioausgabe abgeschlossen.",
        "playback_error": "Fehler beim Abspielen des Audios: {error}"
    },
    "en": {
        "speak_now": "Speak now...",
        "recording": "Recording... (Speak or press Ctrl+C to stop)",
        "auto_stopped": "Recording automatically stopped after silence.",
        "manual_stopped": "Recording manually stopped.",
        "recording_saved": "Recording finished and saved to {}.",
        "no_recording": "No recording made.",
        "enter_answer": "Please enter your answer:",
        "transcribing": "Transcribing audio...",
        "transcription": "Transcription: \"{}\"",
        "confirm_transcription": "Is this transcription correct? (Y/N, or edit the text):",
        "rejected": "Transcription rejected. Please enter your answer:",
        "recording_error": "Error during voice recording: {error}",
        "fallback": "Falling back to text input...",
        "temp_file_error": "Warning: Could not delete temporary file: {error}",
        "generating_speech": "Generating speech output...",
        "playing_audio": "Playing audio...",
        "audio_error": "Error in speech output: {error}",
        "playback_complete": "Audio playback complete.",
        "playback_error": "Error playing audio: {error}"
    }
}

# Funktion für einfache Textübersetzung
def translate(key: str, **kwargs) -> str:
    """
    Get translated text based on the configured language.
    
    Args:
        key: Translation key
        **kwargs: Format parameters
        
    Returns:
        Translated string
    """
    # Default to English if language not supported
    if LANGUAGE not in TRANSLATIONS:
        lang = "en"
    else:
        lang = LANGUAGE
        
    translation = TRANSLATIONS[lang].get(key, key)
    
    if kwargs:
        try:
            return translation.format(**kwargs)
        except KeyError as e:
            print(f"Warnung: Fehlender Parameter {e} in Übersetzung '{key}' für Sprache '{lang}'")
            return translation
    return translation

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
        
        # Ensure temp directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        
    def get_input(self, prompt: str) -> str:
        """
        Get voice input from the user using OpenAI's speech-to-text.
        
        Args:
            prompt: The prompt to show to the user
            
        Returns:
            The transcribed user's input as a string
        """
        print(f"{prompt} ({translate('speak_now')})")
        
        # Create a unique filename in our temp directory
        filename = f"voice_input_{uuid.uuid4().hex}.wav"
        temp_filepath = os.path.join(TEMP_DIR, filename)
        
        try:
            # Import here to avoid dependencies if voice is not used
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            
            # Recording settings
            fs = 44100  # Sample rate (Hz)
            channels = 1  # Mono recording
            recording_duration = 600  # Maximum recording duration in seconds (10 Minuten)
            
            # Visual indicator for recording
            print("\n[", end="", flush=True)
            
            # Record audio with a dynamic duration
            recording = []
            silence_threshold = 0.01
            silence_duration = 0  # Duration of silence in samples
            max_silence_duration = 8 * fs  # 8 seconds of silence to stop
            
            print(translate('recording'))
            
            # Start recording
            with sd.InputStream(samplerate=fs, channels=channels, callback=None) as stream:
                remaining_frames = recording_duration * fs
                
                while remaining_frames > 0:
                    # Read a small chunk of audio
                    frames = min(remaining_frames, int(fs * 0.1))  # 0.1 second chunks
                    data, overflowed = stream.read(frames)
                    if overflowed:
                        print("x", end="", flush=True)
                    else:
                        print("=", end="", flush=True)
                    
                    recording.append(data.copy())
                    remaining_frames -= frames
                    
                    # Check for silence to auto-stop
                    if np.max(np.abs(data)) < silence_threshold:
                        silence_duration += len(data)
                        if silence_duration >= max_silence_duration:
                            print("]")
                            print(translate('auto_stopped'))
                            break
                    else:
                        silence_duration = 0
                    
                    # Allow manual stop with keyboard
                    try:
                        if os.name == 'nt':  # For Windows
                            import msvcrt
                            if msvcrt.kbhit() and msvcrt.getch() == b'\x03':  # Ctrl+C
                                print("]")
                                print(translate('manual_stopped'))
                                break
                        # For Unix-based systems, detection is more complex
                        # and would need a separate thread, so we'll rely on Ctrl+C exception
                    except:
                        pass
            
            # Concatenate all chunks
            if recording:
                recording_array = np.vstack(recording)
                # Save as WAV file
                sf.write(temp_filepath, recording_array, fs)
                print(f"\n{translate('recording_saved', **{'{}': temp_filepath})}")
            else:
                print(f"\n{translate('no_recording')}")
                return input(f"{translate('enter_answer')} ").strip()
            
            # Transcribe with OpenAI
            print(translate('transcribing'))
            with open(temp_filepath, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            transcribed_text = transcript.text
            
            # Show transcription and get confirmation
            print(f"\n{translate('transcription', **{'{}': transcribed_text})}")
            
            confirmation = input(f"\n{translate('confirm_transcription')} ").strip()
            
            yes_responses = ["j", "ja", "y", "yes", ""]
            no_responses = ["n", "nein", "no"]
            
            if LANGUAGE == "de":
                yes_responses = ["j", "ja", "y", "yes", ""]
                no_responses = ["n", "nein", "no"]
            else:
                yes_responses = ["y", "yes", "j", "ja", ""]
                no_responses = ["n", "no", "nein"]
                
            if confirmation.lower() in yes_responses:
                return transcribed_text
            elif confirmation.lower() in no_responses:
                print(translate('rejected'))
                return input("> ").strip()
            else:
                # User edited the text
                return confirmation
        
        except KeyboardInterrupt:
            print(f"\n{translate('manual_stopped')}")
            return input(f"{translate('enter_answer')} ").strip()
        except Exception as e:
            print(f"\n{translate('recording_error', **{'{}': str(e)})}")
            print(translate('fallback'))
            return input(f"{translate('enter_answer')} ").strip()
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception as e:
                    print(f"{translate('temp_file_error', **{'{}': str(e)})}")


class VoiceOutputHandler(OutputHandler):
    """Voice-based output handler using OpenAI's text-to-speech API."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """
        Initialize the voice output handler.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.voice = "nova"  # Default voice, can be: alloy, echo, fable, onyx, nova, shimmer
        self.should_speak = True
        
        # Ensure temp directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Initialize pygame mixer
        self.pygame_initialized = False
        try:
            import pygame
            # Initialize pygame modules but not everything
            pygame.init()
            # Initialize the mixer separately with specific settings
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            self.pygame_initialized = True
            print("Pygame mixer erfolgreich initialisiert.")
        except Exception as e:
            print(f"Warnung: Konnte pygame.mixer nicht initialisieren: {e}")
            print("Versuche alternative Audiowiedergabe...")
            
            # Try to initialize simpleaudio as fallback
            try:
                import simpleaudio
                self.simpleaudio_available = True
                print("SimpleAudio als Fallback verfügbar.")
            except ImportError:
                self.simpleaudio_available = False
                print("SimpleAudio nicht verfügbar.")
                
            # If simpleaudio fails, try pydub
            if not self.simpleaudio_available:
                try:
                    from pydub import AudioSegment
                    from pydub.playback import play
                    self.pydub_available = True
                    print("Pydub als Fallback verfügbar.")
                except ImportError:
                    self.pydub_available = False
                    print("Pydub nicht verfügbar. Keine Audiowiedergabe möglich.")
        
    def display_output(self, message: str) -> None:
        """
        Display output as synthesized speech using OpenAI's text-to-speech.
        
        Args:
            message: The message to convert to speech
        """
        # Always display the message as text first
        print("\n" + "-" * 80)
        print(message)
        print("-" * 80)
        
        # Short messages don't need voice
        if len(message) < 10:
            return
            
        if not self.should_speak:
            return

        try:
            print(translate("generating_speech"))
            
            # Create directories if they don't exist
            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR)
                
            # Generate a unique filename
            import uuid
            temp_filename = f"voice_output_{uuid.uuid4().hex}.mp3"
            temp_filepath = os.path.join(TEMP_DIR, temp_filename)
            
            # Generate speech using OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=message
            )
            
            # Save to a temporary file
            response.stream_to_file(temp_filepath)
            
            # Play the audio
            print(translate("playing_audio"))
            
            # Try pygame first (blocking playback)
            played = False
            
            try:
                if self.pygame_initialized:
                    import pygame
                    pygame.mixer.music.load(temp_filepath)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    played = True
                    print(translate("playback_complete"))
            except Exception as e:
                print(translate("playback_error", error=str(e)))
                
            # If pygame failed, try simpleaudio
            if not played:
                try:
                    import simpleaudio as sa
                    wave_obj = sa.WaveObject.from_wave_file(temp_filepath)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                    played = True
                    print(translate("playback_complete"))
                except ImportError:
                    print("simpleaudio not available, trying pydub...")
                except Exception as e:
                    print(translate("playback_error", error=str(e)))
                    
            # If simpleaudio failed, try pydub
            if not played:
                try:
                    from pydub import AudioSegment
                    from pydub.playback import play
                    
                    sound = AudioSegment.from_file(temp_filepath, format="mp3")
                    play(sound)
                    played = True
                    print(translate("playback_complete"))
                except ImportError:
                    print("pydub not available")
                except Exception as e:
                    print(translate("playback_error", error=str(e)))
            
            # Give time for the audio file to be completely released before attempting to delete
            import time
            time.sleep(0.5)
            
            # Clean up temporary file - don't raise errors if we can't delete
            try:
                os.remove(temp_filepath)
            except Exception as e:
                print(translate("temp_file_error", error=str(e)))
                
        except Exception as e:
            print(translate("audio_error", error=str(e)))
    
    def _play_audio_pygame(self, file_path: str) -> None:
        """
        Play an audio file using pygame.mixer in a blocking manner.
        
        Args:
            file_path: Path to the MP3 audio file
        """
        try:
            import pygame
            
            # Re-initialize mixer if needed
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            
            # Load and play the audio
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)  # Check every 100ms if playback has finished
                
            # Add a small delay to ensure complete playback
            time.sleep(0.5)
            
            print(f"{translate('playback_complete')}")
            
        except Exception as e:
            print(f"{translate('playback_error', **{'{}': str(e)})}")
            print(f"  Pfad: {file_path}")
    
    def _play_audio_simpleaudio(self, file_path: str) -> None:
        """
        Play an audio file using simpleaudio.
        
        Args:
            file_path: Path to the audio file
        """
        try:
            import simpleaudio as sa
            from pydub import AudioSegment
            
            # Convert MP3 to WAV in memory for simpleaudio
            sound = AudioSegment.from_file(file_path, format="mp3")
            
            # Export to a temporary WAV file
            temp_wav = os.path.join(TEMP_DIR, f"temp_wav_{uuid.uuid4().hex}.wav")
            sound.export(temp_wav, format="wav")
            
            # Play using simpleaudio
            wave_obj = sa.WaveObject.from_wave_file(temp_wav)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            print(f"{translate('playback_complete')}")
            
            # Clean up temporary WAV
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
        except Exception as e:
            print(f"{translate('playback_error', **{'{}': str(e)})}")
    
    def _play_audio_pydub(self, file_path: str) -> None:
        """
        Play an audio file using pydub (platform-specific backends).
        
        Args:
            file_path: Path to the audio file
        """
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            
            # Load and play the audio file
            sound = AudioSegment.from_file(file_path, format="mp3")
            play(sound)
            print(f"{translate('playback_complete')}")
            
        except Exception as e:
            print(f"{translate('playback_error', **{'{}': str(e)})}")


def get_input_handler() -> InputHandler:
    """
    Factory function to get the appropriate input handler based on configuration.
    
    Returns:
        An instance of InputHandler
    """
    if VOICE_INPUT_ENABLED:
        try:
            return VoiceInputHandler()
        except Exception as e:
            print(f"Fehler beim Initialisieren des VoiceInputHandlers: {e}")
            print("Fallback auf TextInputHandler.")
            return TextInputHandler()
    else:
        return TextInputHandler()


def get_output_handler() -> OutputHandler:
    """
    Factory function to get the appropriate output handler based on configuration.
    
    Returns:
        An instance of OutputHandler
    """
    if VOICE_OUTPUT_ENABLED:
        try:
            return VoiceOutputHandler()
        except Exception as e:
            print(f"Fehler beim Initialisieren des VoiceOutputHandlers: {e}")
            print("Fallback auf TextOutputHandler.")
            return TextOutputHandler()
    else:
        return TextOutputHandler()
