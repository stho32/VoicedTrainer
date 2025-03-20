"""
Trainer module for the VoicedTrainer application.

This module handles the interactive training session with the user.
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI

from voiced_trainer.config import (
    PROCESSED_DATA_DIR,
    QUESTIONS_PER_TOPIC,
    OPENAI_API_KEY,
    OPENAI_MODEL
)
from voiced_trainer.io_handlers import InputHandler, OutputHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class VoiceTrainer:
    """
    Handles the interactive training session with the user.
    """
    
    def __init__(
        self, 
        input_handler: InputHandler, 
        output_handler: OutputHandler,
        api_key: str = OPENAI_API_KEY
    ):
        """
        Initialize the trainer.
        
        Args:
            input_handler: Handler for user input
            output_handler: Handler for output to the user
            api_key: OpenAI API key
        """
        self.input_handler = input_handler
        self.output_handler = output_handler
        self.client = OpenAI(api_key=api_key)
        
    def _load_topics(self) -> List[Dict[str, str]]:
        """
        Load all processed topic files.
        
        Returns:
            List of topic dictionaries
        """
        topics = []
        topic_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.startswith("topic_") and f.endswith(".json")]
        
        for file_name in topic_files:
            file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    topic = json.load(f)
                topics.append(topic)
            except Exception as e:
                logger.error(f"Error loading topic file {file_path}: {e}")
        
        return topics
    
    def _generate_questions_for_topic(self, topic: Dict[str, str], num_questions: int = QUESTIONS_PER_TOPIC) -> List[Dict[str, Any]]:
        """
        Generate new questions for a specific topic.
        
        Args:
            topic: Topic dictionary with 'title' and 'content'
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Generating {num_questions} questions for topic '{topic['title']}'...")
        
        question_prompt = (
            f"Based on the following topic about '{topic['title']}', create {num_questions} thought-provoking "
            f"questions that would test understanding and critical thinking. For each question, also provide a "
            f"brief guide on what a good answer should include. Format the response so that each question is "
            f"clearly separated."
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
            questions = []
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
                        "topic_title": topic["title"],
                        "question": question,
                        "answer_guide": guide
                    })
                    
                    if len(questions) >= num_questions:
                        break
            
            return questions[:num_questions]  # Ensure we have exactly num_questions
            
        except Exception as e:
            logger.error(f"Error generating questions for topic '{topic['title']}': {e}")
            return []
    
    def _evaluate_answer(self, question: Dict[str, Any], user_answer: str) -> str:
        """
        Evaluate the user's answer using OpenAI.
        
        Args:
            question: Question dictionary
            user_answer: User's answer as a string
            
        Returns:
            Feedback on the user's answer
        """
        evaluation_prompt = (
            f"You are a knowledgeable and supportive tutor. Evaluate the user's answer to the following question "
            f"about {question['topic_title']}:\n\n"
            f"Question: {question['question']}\n\n"
            f"A good answer should include: {question['answer_guide']}\n\n"
            f"User's answer: {user_answer}\n\n"
            f"Provide constructive feedback on the answer, highlighting strengths and areas for improvement. "
            f"Be encouraging but honest. Include at least one follow-up question to deepen understanding. "
            f"Format your response in a conversational, tutoring style."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable and supportive tutor evaluating a student's answer."},
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return "I couldn't properly evaluate your answer. Let's move to the next question."
    
    def run_interactive_session(self) -> None:
        """
        Run an interactive training session with the user.
        """
        # Load the topics
        topics = self._load_topics()
        if not topics:
            self.output_handler.display_output("No topics found. Please ensure preprocessing has been completed.")
            return
        
        # Shuffle the topics
        random.shuffle(topics)
        
        self.output_handler.display_output(
            "Welcome to VoicedTrainer!\n\n"
            "I'll ask you questions about various topics from the material. "
            "Try to answer as completely as you can.\n"
            "Type 'exit' at any time to end the session."
        )
        
        # Main training loop
        for topic in topics:
            self.output_handler.display_output(f"\n\nNew Topic: {topic['title']}\n")
            
            # Generate questions for this topic
            questions = self._generate_questions_for_topic(topic)
            if not questions:
                self.output_handler.display_output(f"Could not generate questions for topic '{topic['title']}'. Skipping.")
                continue
            
            # Ask questions
            for i, question in enumerate(questions, 1):
                self.output_handler.display_output(f"\nQuestion {i}: {question['question']}")
                
                # Get user's answer
                user_answer = self.input_handler.get_input("Your answer:")
                
                # Check for exit command
                if user_answer.lower() == "exit":
                    self.output_handler.display_output("\nThank you for using VoicedTrainer. Goodbye!")
                    return
                
                # Evaluate the answer
                feedback = self._evaluate_answer(question, user_answer)
                self.output_handler.display_output(f"\nFeedback: {feedback}")
                
                # Optional follow-up after feedback
                follow_up = self.input_handler.get_input("Do you have any thoughts on this feedback? (Press Enter to continue)")
                if follow_up.lower() == "exit":
                    self.output_handler.display_output("\nThank you for using VoicedTrainer. Goodbye!")
                    return
            
            # After completing all questions for a topic
            continue_session = self.input_handler.get_input("\nWould you like to continue to the next topic? (yes/no)")
            if continue_session.lower() not in ["yes", "y"]:
                self.output_handler.display_output("\nThank you for using VoicedTrainer. Goodbye!")
                return
        
        # End of all topics
        self.output_handler.display_output(
            "\nCongratulations! You've completed all the topics. "
            "\nThank you for using VoicedTrainer. Goodbye!"
        )
