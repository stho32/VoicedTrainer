"""
Main entry point for the VoicedTrainer application.

This application processes a text file, extracts topics and questions,
and provides an interactive voice-based training experience.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from voiced_trainer.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    PREPROCESSED_LOCK_FILE,
    NUM_TOPICS,
    QUESTIONS_PER_TOPIC,
    VOICE_INPUT_ENABLED,
    VOICE_OUTPUT_ENABLED
)
from voiced_trainer.preprocessor import preprocess_data
from voiced_trainer.trainer import VoiceTrainer
from voiced_trainer.io_handlers import get_input_handler, get_output_handler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """Ensure all necessary directories exist."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)


def find_text_file():
    """Find a suitable text file in the data directory."""
    text_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not text_files:
        logger.error(f"No text files found in {DATA_DIR}")
        return None
    return text_files[0]


def main():
    """
    Main function to run the VoicedTrainer application.
    """
    parser = argparse.ArgumentParser(description="VoicedTrainer - Interactive voice-based training application")
    parser.add_argument("--topics", type=int, default=NUM_TOPICS, help="Number of topics to extract from the material")
    parser.add_argument("--questions", type=int, default=QUESTIONS_PER_TOPIC, help="Number of questions per topic")
    parser.add_argument("--voice-input", action="store_true", help="Enable voice input")
    parser.add_argument("--voice-output", action="store_true", help="Enable voice output")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing even if not done")
    parser.add_argument("--force-preprocessing", action="store_true", help="Force preprocessing even if already done")
    args = parser.parse_args()

    # Update config values if specified in command line
    # Wir verwenden die Config-Variablen direkt, ohne sie als global zu deklarieren
    import voiced_trainer.config as config
    config.NUM_TOPICS = args.topics
    config.QUESTIONS_PER_TOPIC = args.questions
    config.VOICE_INPUT_ENABLED = args.voice_input
    config.VOICE_OUTPUT_ENABLED = args.voice_output

    # Print welcome message
    print("\n" + "=" * 60)
    print("Welcome to VoicedTrainer".center(60))
    print("=" * 60 + "\n")

    # Make sure directories exist
    ensure_directories()

    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("\nWARNING: OpenAI API key not found in environment variables.")
        print("Please set the OPENAI_API_KEY environment variable to use this application.")
        print("You can set it temporarily with:")
        print("  On Windows: set OPENAI_API_KEY=your_api_key")
        print("  On Unix/MacOS: export OPENAI_API_KEY=your_api_key\n")
        
        # Ask if user wants to continue without API key
        response = input("Do you want to continue anyway? This will likely cause errors. (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Exiting.")
            sys.exit(1)

    # Find a text file to process
    if not os.path.exists(PREPROCESSED_LOCK_FILE) or args.force_preprocessing:
        text_file = find_text_file()
        if not text_file:
            print(f"Error: No text files found in {DATA_DIR}")
            print("Please add a text file to the data directory and try again.")
            sys.exit(1)
        
        print(f"Found text file: {text_file}")
        
        # Preprocess the data if needed
        if args.force_preprocessing or not os.path.exists(PREPROCESSED_LOCK_FILE):
            print("\nPreprocessing text data...")
            if preprocess_data(text_file):
                print("Preprocessing completed successfully!")
            else:
                print("Preprocessing failed or was skipped.")
                if not args.skip_preprocessing and not os.path.exists(PREPROCESSED_LOCK_FILE):
                    print("Exiting.")
                    sys.exit(1)
    elif args.skip_preprocessing:
        print("Skipping preprocessing as requested.")
    else:
        print("Preprocessing already completed. Skipping.")

    # Initialize I/O handlers
    input_handler = get_input_handler()
    output_handler = get_output_handler()
    
    # Start interactive training session
    print("\nStarting interactive training session...\n")
    trainer = VoiceTrainer(input_handler, output_handler)
    trainer.run_interactive_session()


if __name__ == "__main__":
    main()
