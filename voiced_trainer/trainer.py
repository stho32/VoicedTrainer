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
    OPENAI_MODEL,
    LANGUAGE
)
from voiced_trainer.io_handlers import InputHandler, OutputHandler, translate

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
    
    def _generate_topic_introduction(self, topic: Dict[str, str]) -> str:
        """
        Generate an introduction to a topic using OpenAI.
        
        Args:
            topic: Topic dictionary with 'title' and 'content'
            
        Returns:
            Introduction text
        """
        logger.info(f"Generating introduction for topic '{topic['title']}'")
        
        # Anpassen der Systemrolle je nach Spracheinstellung
        system_role = "Du bist ein hilfreicher Assistent, der Bildungsthemen erklärt." if LANGUAGE == "de" else \
                      "You are a helpful assistant that explains educational topics."
        
        # Anpassen des Prompts je nach Spracheinstellung
        introduction_prompt = ""
        if LANGUAGE == "de":
            introduction_prompt = (
                f"Erstelle eine kurze, motivierende Einführung (ca. 3-4 Sätze) zum Thema '{topic['title']}'. "
                f"Die Einführung sollte das Wesentliche des Themas erfassen und den Lernenden neugierig machen. "
                f"Verwende einen freundlichen, einladenden Ton. Basiere deine Einführung auf folgendem Inhalt:\n\n"
                f"{topic['content'][:500]}..."  # Beschränke den Inhalt, um den Prompt kurz zu halten
            )
        else:
            introduction_prompt = (
                f"Create a short, motivating introduction (about 3-4 sentences) for the topic '{topic['title']}'. "
                f"The introduction should capture the essence of the topic and make the learner curious. "
                f"Use a friendly, inviting tone. Base your introduction on the following content:\n\n"
                f"{topic['content'][:500]}..."  # Limit content to keep the prompt short
            )
            
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": introduction_prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating introduction for topic '{topic['title']}': {e}")
            # Fallback-Einführungen je nach Sprache
            if LANGUAGE == "de":
                return f"Lass uns über das Thema '{topic['title']}' sprechen."
            else:
                return f"Let's talk about the topic '{topic['title']}'."
    
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
        
        # Anpassen des Prompts je nach Spracheinstellung
        system_role = "Du bist ein hilfreicher Assistent, der Bildungsfragen erstellt." if LANGUAGE == "de" else \
                      "You are a helpful assistant that creates educational assessment questions."
        
        question_prompt = ""
        if LANGUAGE == "de":
            question_prompt = (
                f"Basierend auf dem folgenden Thema über '{topic['title']}', erstelle {num_questions} zum Nachdenken "
                f"anregende Fragen, die das Verständnis und kritisches Denken testen. Füge zu jeder Frage eine "
                f"kurze Anleitung hinzu, was eine gute Antwort enthalten sollte. Formatiere die Antwort so, dass "
                f"jede Frage klar voneinander getrennt ist."
            )
        else:
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
                    {"role": "system", "content": system_role},
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
                    if ":" in question and question.split(":", 1)[0].strip().lower() in ["question", "q", "frage", "f"]:
                        question = question.split(":", 1)[1].strip()
                    elif question[0].isdigit() and ". " in question:
                        question = question.split(". ", 1)[1].strip()
                    
                    guide = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Clean up the guide
                    guide_prefixes = ["guide:", "answer:", "good answer:", "antwort:", "gute antwort:", "anleitung:"]
                    for prefix in guide_prefixes:
                        if guide.lower().startswith(prefix):
                            guide = guide.split(":", 1)[1].strip()
                            break
                    
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
    
    def _evaluate_answer(self, question: Dict[str, Any], user_answer: str) -> Dict[str, str]:
        """
        Evaluate the user's answer using OpenAI.
        
        Args:
            question: Question dictionary
            user_answer: User's answer as a string
            
        Returns:
            Dictionary with feedback and a list of follow-up questions
        """
        # Anpassen des Prompts je nach Spracheinstellung
        system_role = "Du bist ein kenntnisreicher und unterstützender Tutor, der eine Schülerantwort bewertet." if LANGUAGE == "de" else \
                      "You are a knowledgeable and supportive tutor evaluating a student's answer."
        
        evaluation_prompt = ""
        if LANGUAGE == "de":
            evaluation_prompt = (
                f"Als Tutor bewerte die Antwort des Lernenden auf die folgende Frage zum Thema {question['topic_title']}:\n\n"
                f"Frage: {question['question']}\n\n"
                f"Eine gute Antwort sollte enthalten: {question['answer_guide']}\n\n"
                f"Antwort des Lernenden: {user_answer}\n\n"
                f"Gib konstruktives Feedback zur Antwort, hebe Stärken und Verbesserungsbereiche hervor. Sei ermutigend, "
                f"aber ehrlich. Bereite drei Nachfragen vor, um das Verständnis zu vertiefen. Teile deine Antwort in "
                f"zwei Abschnitte: 1) Feedback und 2) Nachfragen. Markiere die Nachfragen deutlich mit der Überschrift "
                f"'Nachfragen:'. Wenn du Nachholbedarf erkennst, teile die komplexe Frage in einfachere Teilfragen auf."
            )
        else:
            evaluation_prompt = (
                f"As a tutor, evaluate the learner's answer to the following question about {question['topic_title']}:\n\n"
                f"Question: {question['question']}\n\n"
                f"A good answer should include: {question['answer_guide']}\n\n"
                f"Learner's answer: {user_answer}\n\n"
                f"Provide constructive feedback on the answer, highlighting strengths and areas for improvement. "
                f"Be encouraging but honest. Prepare three follow-up questions to deepen understanding. Divide your "
                f"response into two sections: 1) Feedback and 2) Follow-up questions. Mark the follow-up questions "
                f"clearly with the heading 'Follow-up questions:'. If you identify learning gaps, break down the complex "
                f"question into simpler sub-questions."
            )
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            
            full_response = response.choices[0].message.content
            
            # Aufteilen in Feedback und Nachfragen
            parts = {}
            
            # Suche nach "Nachfragen:" oder "Follow-up questions:"
            follow_up_headers = ["nachfragen:", "follow-up questions:", "follow up questions:"]
            
            for header in follow_up_headers:
                if header in full_response.lower():
                    split_text = full_response.lower().split(header, 1)
                    parts["feedback"] = split_text[0].strip()
                    parts["follow_up_questions"] = split_text[1].strip()
                    break
            
            # Wenn keine klare Aufteilung gefunden wurde, verwende alles als Feedback
            if not parts:
                parts["feedback"] = full_response
                parts["follow_up_questions"] = ""
                
            return parts
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            # Standardantworten je nach Sprache
            if LANGUAGE == "de":
                return {
                    "feedback": "Ich konnte deine Antwort nicht richtig bewerten. Lass uns zur nächsten Frage übergehen.",
                    "follow_up_questions": ""
                }
            else:
                return {
                    "feedback": "I couldn't properly evaluate your answer. Let's move to the next question.",
                    "follow_up_questions": ""
                }
    
    def _handle_follow_up_questions(self, follow_up_text: str) -> List[str]:
        """
        Parse follow-up questions from text and handle them one by one.
        
        Args:
            follow_up_text: Text containing follow-up questions
            
        Returns:
            List of question/answer pairs
        """
        # Versuche, einzelne Fragen aus dem Text zu extrahieren
        questions = []
        lines = follow_up_text.split('\n')
        
        current_question = ""
        for line in lines:
            line = line.strip()
            # Versuche, nummerierte Fragen oder mit Sternen markierte Fragen zu finden
            if (line and (line[0].isdigit() and ". " in line[:5]) or 
                line.startswith("- ") or line.startswith("* ")):
                if current_question:
                    questions.append(current_question)
                current_question = line
            elif current_question and line:
                current_question += " " + line
            
        if current_question:
            questions.append(current_question)
        
        # Wenn keine Fragen gefunden wurden, aber Text vorhanden ist, behandle den ganzen Text als eine Frage
        if not questions and follow_up_text.strip():
            questions = [follow_up_text.strip()]
            
        # Begrenze auf maximal 3 Fragen
        return questions[:3]
    
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
        
        # Angepasste Begrüßung je nach Spracheinstellung
        if LANGUAGE == "de":
            welcome_message = (
                "Willkommen bei VoicedTrainer!\n\n"
                "Ich werde dir Fragen zu verschiedenen Themen stellen. "
                "Versuche, so vollständig wie möglich zu antworten.\n"
                "Gib 'beenden' ein, um die Sitzung jederzeit zu beenden."
            )
        else:
            welcome_message = (
                "Welcome to VoicedTrainer!\n\n"
                "I'll ask you questions about various topics from the material. "
                "Try to answer as completely as you can.\n"
                "Type 'exit' at any time to end the session."
            )
            
        self.output_handler.display_output(welcome_message)
        
        # Exit commands je nach Sprache
        exit_commands = ["exit", "quit", "beenden", "ende", "stop", "schließen"]
        
        # Main training loop
        for topic in topics:
            # Generate introduction for this topic
            intro = self._generate_topic_introduction(topic)
            
            # Themenüberschrift je nach Sprache
            if LANGUAGE == "de":
                topic_header = f"\n\nNeues Thema: {topic['title']}\n"
            else:
                topic_header = f"\n\nNew Topic: {topic['title']}\n"
                
            self.output_handler.display_output(f"{topic_header}{intro}")
            
            # Generate questions for this topic
            questions = self._generate_questions_for_topic(topic)
            if not questions:
                if LANGUAGE == "de":
                    skip_message = f"Konnte keine Fragen zum Thema '{topic['title']}' generieren. Überspringe."
                else:
                    skip_message = f"Could not generate questions for topic '{topic['title']}'. Skipping."
                self.output_handler.display_output(skip_message)
                continue
            
            # Zähler für Nachfragen
            follow_up_counter = 0
            question_counter = 0
            
            # Ask questions
            while question_counter < len(questions):
                question = questions[question_counter]
                
                # Frage-Header je nach Sprache
                if LANGUAGE == "de":
                    question_header = f"\nFrage {question_counter + 1}: "
                else:
                    question_header = f"\nQuestion {question_counter + 1}: "
                    
                self.output_handler.display_output(f"{question_header}{question['question']}")
                
                # Get user's answer
                user_answer = self.input_handler.get_input("Your answer:")
                
                # Check for exit command
                if user_answer.lower() in exit_commands:
                    if LANGUAGE == "de":
                        goodbye = "\nVielen Dank für die Nutzung von VoicedTrainer. Auf Wiedersehen!"
                    else:
                        goodbye = "\nThank you for using VoicedTrainer. Goodbye!"
                    self.output_handler.display_output(goodbye)
                    return
                
                # Evaluate the answer
                evaluation = self._evaluate_answer(question, user_answer)
                
                # Feedback-Header je nach Sprache
                if LANGUAGE == "de":
                    feedback_header = "\nFeedback: "
                else:
                    feedback_header = "\nFeedback: "
                    
                self.output_handler.display_output(f"{feedback_header}{evaluation['feedback']}")
                
                # Handle follow-up questions if there are any
                follow_up_questions = self._handle_follow_up_questions(evaluation.get("follow_up_questions", ""))
                
                for follow_up in follow_up_questions:
                    # Erhöhe den Nachfragen-Zähler
                    follow_up_counter += 1
                    
                    # Formatiere die Nachfrage je nach Sprache
                    if LANGUAGE == "de":
                        follow_up_prefix = "\nNachfrage: "
                    else:
                        follow_up_prefix = "\nFollow-up question: "
                        
                    self.output_handler.display_output(f"{follow_up_prefix}{follow_up}")
                    
                    # Get user's answer to the follow-up
                    follow_up_answer = self.input_handler.get_input("Your answer:")
                    
                    # Check for exit command
                    if follow_up_answer.lower() in exit_commands:
                        if LANGUAGE == "de":
                            goodbye = "\nVielen Dank für die Nutzung von VoicedTrainer. Auf Wiedersehen!"
                        else:
                            goodbye = "\nThank you for using VoicedTrainer. Goodbye!"
                        self.output_handler.display_output(goodbye)
                        return
                    
                    # Kurzes Feedback zur Nachfrage
                    if LANGUAGE == "de":
                        self.output_handler.display_output("\nDanke für deine Antwort.")
                    else:
                        self.output_handler.display_output("\nThank you for your answer.")
                    
                    # Nach jeder 3-4 Nachfragen fragen, ob zum nächsten Thema gewechselt werden soll
                    if follow_up_counter >= random.randint(3, 4):
                        if LANGUAGE == "de":
                            continue_prompt = "\nMöchtest du zum nächsten Thema wechseln oder weiter über dieses Thema sprechen? (nächstes/weiter)"
                        else:
                            continue_prompt = "\nWould you like to move to the next topic or continue with this one? (next/continue)"
                            
                        decision = self.input_handler.get_input(continue_prompt)
                        
                        # Check for exit command
                        if decision.lower() in exit_commands:
                            if LANGUAGE == "de":
                                goodbye = "\nVielen Dank für die Nutzung von VoicedTrainer. Auf Wiedersehen!"
                            else:
                                goodbye = "\nThank you for using VoicedTrainer. Goodbye!"
                            self.output_handler.display_output(goodbye)
                            return
                        
                        # Entscheidung verarbeiten
                        next_topic_keywords = ["nächstes", "next", "ja", "yes", "y", "j"]
                        if any(keyword in decision.lower() for keyword in next_topic_keywords):
                            # Zum nächsten Thema springen
                            question_counter = len(questions)  # Das beendet die aktuelle while-Schleife
                            break  # Aus der Nachfragen-Schleife ausbrechen
                        else:
                            # Nachfragenzähler zurücksetzen
                            follow_up_counter = 0
                
                # Nächste Frage
                question_counter += 1
            
            # Nach Abschluss aller Fragen für ein Thema
            if LANGUAGE == "de":
                continue_prompt = "\nMöchtest du mit dem nächsten Thema fortfahren? (ja/nein)"
            else:
                continue_prompt = "\nWould you like to continue to the next topic? (yes/no)"
                
            continue_session = self.input_handler.get_input(continue_prompt)
            
            yes_responses = ["ja", "j", "yes", "y"]
            if not any(resp in continue_session.lower() for resp in yes_responses):
                if LANGUAGE == "de":
                    goodbye = "\nVielen Dank für die Nutzung von VoicedTrainer. Auf Wiedersehen!"
                else:
                    goodbye = "\nThank you for using VoicedTrainer. Goodbye!"
                self.output_handler.display_output(goodbye)
                return
        
        # Ende aller Themen
        if LANGUAGE == "de":
            completion_message = (
                "\nHerzlichen Glückwunsch! Du hast alle Themen abgeschlossen. "
                "\nVielen Dank für die Nutzung von VoicedTrainer. Auf Wiedersehen!"
            )
        else:
            completion_message = (
                "\nCongratulations! You've completed all the topics. "
                "\nThank you for using VoicedTrainer. Goodbye!"
            )
            
        self.output_handler.display_output(completion_message)
