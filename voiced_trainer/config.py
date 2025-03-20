"""
Configuration settings for the VoicedTrainer application.
"""

# Number of topics to extract from the source material
NUM_TOPICS = 10

# Number of questions per topic file during interactive session
QUESTIONS_PER_TOPIC = 5

# Data directory paths
import os
import pathlib

# Repository root directory
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "voiced_trainer", "data")

# Preprocessed lock file to indicate preprocessing is complete
PREPROCESSED_LOCK_FILE = os.path.join(PROCESSED_DATA_DIR, "preprocessed.lock")

# Temp directory
TEMP_DIR = os.path.join(BASE_DIR, "temp")  # Verzeichnis für temporäre Dateien

# OpenAI API settings - these should be loaded from environment variables in production
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"  # Model for text generation

# App settings
LANGUAGE = "de"  # Language setting: 'de' for German, 'en' for English

# Voice settings
VOICE_INPUT_ENABLED = True
VOICE_OUTPUT_ENABLED = True
