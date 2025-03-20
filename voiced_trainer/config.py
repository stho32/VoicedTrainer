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
ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "voiced_trainer", "data")

# Preprocessed lock file to indicate preprocessing is complete
PREPROCESSED_LOCK_FILE = os.path.join(PROCESSED_DATA_DIR, "preprocessed.lock")

# OpenAI API settings - these should be loaded from environment variables in production
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"  # Model for text generation

# Voice settings
VOICE_INPUT_ENABLED = False
VOICE_OUTPUT_ENABLED = False
