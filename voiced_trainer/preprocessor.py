"""
Text preprocessor for VoicedTrainer.

This module handles the preprocessing of text files to extract topics and generate questions.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from tqdm import tqdm

from voiced_trainer.config import (
    DATA_DIR, 
    PROCESSED_DATA_DIR, 
    PREPROCESSED_LOCK_FILE,
    NUM_TOPICS,
    OPENAI_API_KEY,
    OPENAI_MODEL
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Handles preprocessing of text documents for the VoicedTrainer.
    """
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """
        Initialize the preprocessor.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        
        # Ensure directories exist
        Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
        
    def _is_already_preprocessed(self) -> bool:
        """Check if preprocessing has already been done."""
        return os.path.exists(PREPROCESSED_LOCK_FILE)
    
    def _create_lock_file(self) -> None:
        """Create a lock file to indicate preprocessing is complete."""
        with open(PREPROCESSED_LOCK_FILE, "w") as f:
            f.write("Preprocessing completed")
        logger.info(f"Created lock file at {PREPROCESSED_LOCK_FILE}")
    
    def _read_text_file(self, file_path: str) -> str:
        """
        Read content from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            The text content as a string
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 4000) -> List[str]:
        """
        Split text into manageable chunks.
        
        Args:
            text: The text to split
            chunk_size: Approximate size of each chunk in characters
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk and start a new one
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _extract_topics(self, text_chunks: List[str], num_topics: int = NUM_TOPICS) -> List[Dict[str, str]]:
        """
        Extract main topics from text chunks using OpenAI.
        
        Args:
            text_chunks: List of text chunks
            num_topics: Number of topics to extract
            
        Returns:
            List of topic dictionaries with 'title' and 'content'
        """
        logger.info(f"Extracting {num_topics} topics from text...")
        
        # First, we'll extract potential topic titles from the chunks
        topics_prompt = (
            f"Analyze the following text and identify {num_topics} distinct main topics or concepts. "
            f"For each topic, provide a clear, concise title and a brief one-sentence description. "
            f"Format as a numbered list."
        )
        
        # Join some chunks to get a good overview (limit to avoid token limits)
        sample_text = "\n\n".join(text_chunks[:min(5, len(text_chunks))])
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key topics from text."},
                    {"role": "user", "content": f"{topics_prompt}\n\nTEXT:\n{sample_text}"}
                ]
            )
            
            topics_text = response.choices[0].message.content
            logger.info(f"Extracted topic suggestions:\n{topics_text}")
            
            # Now generate detailed content for each topic
            topics = []
            
            # Extract topic titles (assuming the format is "1. Title: Description")
            topic_titles = []
            for line in topics_text.split("\n"):
                if line.strip() and line[0].isdigit() and ". " in line:
                    title_part = line.split(": ")[0].split(". ")[1] if ": " in line else line.split(". ")[1]
                    topic_titles.append(title_part.strip())
            
            # Ensure we have exactly num_topics
            topic_titles = topic_titles[:num_topics]
            while len(topic_titles) < num_topics:
                topic_titles.append(f"Additional Topic {len(topic_titles) + 1}")
            
            # For each topic, generate detailed content
            for title in tqdm(topic_titles, desc="Generating topic content"):
                content_prompt = (
                    f"Based on the following text, create a comprehensive explanation about the topic '{title}'. "
                    f"Include key concepts, examples, and insights from the text. "
                    f"The content should be detailed enough to serve as learning material (about 500-800 words)."
                )
                
                # Use all chunks for comprehensive content
                complete_text = "\n\n".join(text_chunks)
                
                content_response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates educational content."},
                        {"role": "user", "content": f"{content_prompt}\n\nTEXT:\n{complete_text}"}
                    ]
                )
                
                content = content_response.choices[0].message.content
                topics.append({
                    "title": title,
                    "content": content
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def _generate_questions(self, topics: List[Dict[str, str]], num_questions: int = 10) -> List[Dict[str, Any]]:
        """
        Generate questions for the extracted topics.
        
        Args:
            topics: List of topic dictionaries
            num_questions: Total number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Generating {num_questions} questions from topics...")
        
        questions = []
        
        # Calculate how many questions per topic
        questions_per_topic = max(1, num_questions // len(topics))
        remaining_questions = num_questions - (questions_per_topic * len(topics))
        
        for i, topic in enumerate(tqdm(topics, desc="Generating questions")):
            # Determine number of questions for this topic
            topic_question_count = questions_per_topic
            if remaining_questions > 0:
                topic_question_count += 1
                remaining_questions -= 1
            
            question_prompt = (
                f"Based on the following topic about '{topic['title']}', create {topic_question_count} thought-provoking "
                f"questions that would test understanding and critical thinking. For each question, also provide a "
                f"brief guide on what a good answer should include."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates educational assessment questions."},
                        {"role": "user", "content": f"{question_prompt}\n\nTOPIC CONTENT:\n{topic['content']}"}
                    ]
                )
                
                questions_text = response.choices[0].message.content
                
                # Parse the generated questions and answers
                # This is a simple approach; might need refinement based on actual output format
                question_blocks = questions_text.split("\n\n")
                
                for block in question_blocks:
                    if "?" in block:
                        # Extract question and guide parts
                        parts = block.split("?", 1)
                        question = parts[0].strip() + "?"
                        
                        # Remove any numbering or "Question:" prefix
                        if ":" in question and question.split(":", 1)[0].strip().lower() in ["question", "q"]:
                            question = question.split(":", 1)[1].strip()
                        elif question[0].isdigit() and ". " in question:
                            question = question.split(". ", 1)[1].strip()
                        
                        guide = parts[1].strip() if len(parts) > 1 else ""
                        
                        # Clean up the guide
                        if guide.lower().startswith(("guide:", "answer:", "good answer:")):
                            guide = guide.split(":", 1)[1].strip()
                        
                        questions.append({
                            "topic_id": i,
                            "topic_title": topic["title"],
                            "question": question,
                            "answer_guide": guide
                        })
                        
                        if len(questions) >= num_questions:
                            break
            
            except Exception as e:
                logger.error(f"Error generating questions for topic '{topic['title']}': {e}")
        
        return questions
    
    def _save_processed_data(self, topics: List[Dict[str, str]], questions: List[Dict[str, Any]]) -> None:
        """
        Save processed topics and questions to files.
        
        Args:
            topics: List of topic dictionaries
            questions: List of question dictionaries
        """
        # Save each topic to a separate file
        for i, topic in enumerate(topics):
            topic_file = os.path.join(PROCESSED_DATA_DIR, f"topic_{i+1}.json")
            with open(topic_file, "w", encoding="utf-8") as f:
                json.dump(topic, f, ensure_ascii=False, indent=2)
                
        # Save all questions to a file
        questions_file = os.path.join(PROCESSED_DATA_DIR, "questions.json")
        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(topics)} topics and {len(questions)} questions to {PROCESSED_DATA_DIR}")
    
    def preprocess(self, file_name: str) -> bool:
        """
        Preprocess a text file to extract topics and generate questions.
        
        Args:
            file_name: Name of the file in the data directory
            
        Returns:
            True if preprocessing was done, False if it was skipped or failed
        """
        if self._is_already_preprocessed():
            logger.info("Preprocessing already done. Skipping.")
            return False
        
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found")
            return False
        
        logger.info(f"Starting preprocessing of {file_path}")
        
        # Read the text file
        text = self._read_text_file(file_path)
        if not text:
            return False
        
        # Split into chunks
        chunks = self._split_text_into_chunks(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Extract topics
        topics = self._extract_topics(chunks, NUM_TOPICS)
        if not topics:
            return False
        
        # Generate questions
        questions = self._generate_questions(topics)
        if not questions:
            return False
        
        # Save processed data
        self._save_processed_data(topics, questions)
        
        # Create lock file
        self._create_lock_file()
        
        return True


def preprocess_data(file_name: str) -> bool:
    """
    Preprocess a text file if not already done.
    
    Args:
        file_name: Name of the file in the data directory
        
    Returns:
        True if preprocessing was done, False if skipped or failed
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(file_name)
