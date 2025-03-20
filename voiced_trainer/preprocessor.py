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
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 3000) -> List[str]:
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
    
    def _extract_topics_from_chunks(self, text_chunks: List[str], num_topics: int = NUM_TOPICS) -> List[Dict[str, str]]:
        """
        Extract main topics from text chunks using a hierarchical approach for large documents.
        
        Args:
            text_chunks: List of text chunks
            num_topics: Number of topics to extract
            
        Returns:
            List of topic dictionaries with 'title' and high-level 'content'
        """
        logger.info(f"Extracting {num_topics} topics from text using a hierarchical approach...")
        
        # Step 1: Create summaries of each chunk
        chunk_summaries = []
        
        for i, chunk in enumerate(tqdm(text_chunks, desc="Summarizing chunks")):
            if not chunk.strip():
                continue
                
            summary_prompt = (
                f"Summarize the following text in 2-3 sentences, capturing its key points and main ideas:"
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes text accurately."},
                        {"role": "user", "content": f"{summary_prompt}\n\nTEXT:\n{chunk}"}
                    ]
                )
                
                summary = response.choices[0].message.content
                chunk_summaries.append(summary)
                
            except Exception as e:
                logger.error(f"Error summarizing chunk {i}: {e}")
                # Add a placeholder if summarization fails
                chunk_summaries.append(f"Content from section {i+1}")
        
        # Step 2: Combine summaries into batches to avoid token limits
        combined_summaries = []
        
        batch_size = 10  # Number of summaries per batch
        for i in range(0, len(chunk_summaries), batch_size):
            batch = chunk_summaries[i:i+batch_size]
            combined_summaries.append("\n\n".join(batch))
        
        # Step 3: Extract potential topics from each summary batch
        all_potential_topics = []
        
        for i, summary_batch in enumerate(tqdm(combined_summaries, desc="Analyzing summary batches")):
            topics_prompt = (
                f"Analyze the following text summaries and identify important topics or concepts. "
                f"For each topic, provide a clear, concise title and a brief one-sentence description. "
                f"Format as a numbered list with 'Topic: [title]' and 'Description: [description]' on separate lines."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts key topics from text."},
                        {"role": "user", "content": f"{topics_prompt}\n\nTEXT SUMMARIES:\n{summary_batch}"}
                    ]
                )
                
                topics_text = response.choices[0].message.content
                
                # Parse the topics
                current_topic = None
                current_description = None
                
                for line in topics_text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line contains a topic
                    if line.lower().startswith(("topic:", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")):
                        # Save previous topic if exists
                        if current_topic and current_description:
                            all_potential_topics.append({
                                "title": current_topic,
                                "description": current_description
                            })
                        
                        # Extract new topic
                        if ":" in line:
                            parts = line.split(":", 1)
                            if len(parts) > 1:
                                topic_text = parts[1].strip()
                                if line.lower().startswith("topic:"):
                                    current_topic = topic_text
                                    current_description = None
                                else:
                                    # Handle numbered list with colon
                                    if "topic:" in line.lower():
                                        topic_part = line.lower().split("topic:", 1)[1].strip()
                                        current_topic = topic_part
                                        current_description = None
                            else:
                                current_topic = parts[0].strip()
                                current_description = None
                        else:
                            # Handle numbered list without colon
                            if "." in line:
                                current_topic = line.split(".", 1)[1].strip()
                                current_description = None
                    
                    # Check if line contains a description
                    elif line.lower().startswith("description:") and current_topic:
                        current_description = line.split(":", 1)[1].strip()
                        
                        # If we have both topic and description, save it
                        if current_topic and current_description:
                            all_potential_topics.append({
                                "title": current_topic,
                                "description": current_description
                            })
                            current_topic = None
                            current_description = None
                
                # Add last topic if not added yet
                if current_topic and current_description:
                    all_potential_topics.append({
                        "title": current_topic,
                        "description": current_description
                    })
                    
            except Exception as e:
                logger.error(f"Error extracting topics from summary batch {i}: {e}")
        
        # Step 4: Consolidate and select the final set of topics
        topics_summary = "\n".join([f"- {topic['title']}: {topic['description']}" 
                                  for topic in all_potential_topics[:min(30, len(all_potential_topics))]])
        
        final_topics_prompt = (
            f"Based on the following list of potential topics extracted from a document, "
            f"identify the {num_topics} most significant and representative topics. "
            f"Combine similar topics and ensure diversity of coverage. "
            f"For each final topic, provide a clear, concise title.\n\n"
            f"POTENTIAL TOPICS:\n{topics_summary}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that consolidates topics effectively."},
                    {"role": "user", "content": final_topics_prompt}
                ]
            )
            
            final_topics_text = response.choices[0].message.content
            logger.info(f"Final topics selection:\n{final_topics_text}")
            
            # Parse final topics
            final_topics = []
            current_topic = ""
            
            for line in final_topics_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # Extract topic titles from numbered list or bullet points
                if line[0].isdigit() and ". " in line:
                    current_topic = line.split(". ", 1)[1].strip()
                    if ":" in current_topic:
                        current_topic = current_topic.split(":", 1)[0].strip()
                    final_topics.append({"title": current_topic, "content": ""})
                elif line.startswith("- ") or line.startswith("* "):
                    current_topic = line[2:].strip()
                    if ":" in current_topic:
                        current_topic = current_topic.split(":", 1)[0].strip()
                    final_topics.append({"title": current_topic, "content": ""})
            
            # Ensure we have exactly num_topics
            final_topics = final_topics[:num_topics]
            while len(final_topics) < num_topics:
                topic_index = len(final_topics) + 1
                final_topics.append({"title": f"Topic {topic_index}", "content": ""})
            
            return final_topics
            
        except Exception as e:
            logger.error(f"Error consolidating final topics: {e}")
            
            # Create default topics as fallback
            default_topics = []
            for i in range(num_topics):
                default_topics.append({
                    "title": f"Topic {i+1}",
                    "content": ""
                })
            return default_topics
    
    def _generate_topic_content(self, topic_title: str, text_chunks: List[str]) -> str:
        """
        Generate detailed content for a specific topic by analyzing relevant chunks.
        
        Args:
            topic_title: The title of the topic
            text_chunks: List of all text chunks
            
        Returns:
            Detailed content for the topic
        """
        logger.info(f"Generating content for topic '{topic_title}'...")
        
        # Step 1: Identify chunks relevant to this topic
        relevant_chunks = []
        
        for i, chunk in enumerate(text_chunks):
            relevance_prompt = (
                f"Determine if the following text is relevant to the topic '{topic_title}'. "
                f"Answer with only 'Yes' or 'No'."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant determining text relevance."},
                        {"role": "user", "content": f"{relevance_prompt}\n\nTEXT:\n{chunk}"}
                    ]
                )
                
                answer = response.choices[0].message.content.strip().lower()
                if "yes" in answer:
                    relevant_chunks.append(chunk)
                    
                # Limit the number of relevant chunks to avoid token limits
                if len(relevant_chunks) >= 5:
                    break
                    
            except Exception as e:
                logger.error(f"Error assessing relevance for chunk {i}: {e}")
        
        # If no chunks were found relevant, use the first few chunks as fallback
        if not relevant_chunks and text_chunks:
            relevant_chunks = text_chunks[:min(3, len(text_chunks))]
        
        # Step 2: Generate content based on relevant chunks
        if not relevant_chunks:
            return f"No detailed information available for '{topic_title}'."
        
        # Combine relevant chunks, but ensure we don't exceed token limits
        combined_text = "\n\n".join(relevant_chunks[:3])  # Limit to 3 chunks
        
        content_prompt = (
            f"Based on the following text excerpts, create a comprehensive explanation about the topic '{topic_title}'. "
            f"Include key concepts, examples, and insights from the text. "
            f"The content should be detailed enough to serve as learning material (about 500-800 words)."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates educational content."},
                    {"role": "user", "content": f"{content_prompt}\n\nTEXT EXCERPTS:\n{combined_text}"}
                ]
            )
            
            content = response.choices[0].message.content
            return content
            
        except Exception as e:
            logger.error(f"Error generating content for topic '{topic_title}': {e}")
            return f"Failed to generate detailed content for '{topic_title}' due to an error."
    
    def _extract_topics(self, text_chunks: List[str], num_topics: int = NUM_TOPICS) -> List[Dict[str, str]]:
        """
        Extract main topics from text chunks using OpenAI.
        
        Args:
            text_chunks: List of text chunks
            num_topics: Number of topics to extract
            
        Returns:
            List of topic dictionaries with 'title' and 'content'
        """
        # First, extract topic titles and high-level descriptions
        topics = self._extract_topics_from_chunks(text_chunks, num_topics)
        
        # Then generate detailed content for each topic
        for topic in tqdm(topics, desc="Generating topic content"):
            topic["content"] = self._generate_topic_content(topic["title"], text_chunks)
        
        return topics
    
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
