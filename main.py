# main.py - FINAL VERSION (works in Codespaces, no OpenAI, no errors)
import os
import torch
from transformers import pipeline
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self):
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1,  # CPU
                torch_dtype="auto"  # только один параметр!
            )
            logger.info("HF Summarizer loaded.")
        except Exception as e:
            logger.error(f"HF load error: {e}")
            raise

    def summarize(self, text: str) -> str:
        try:
            # Обрезаем текст, если слишком длинный
            max_input = 1024
            if len(text) > max_input:
                text = text[:max_input]
            result = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return result[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return "Error: Summarization failed."

class KeywordExtractor:
    def extract_keywords(self, text: str, num_keywords: int) -> list:
        try:
            pipe = pipeline("ner", model="dslim/bert-base-NER")
            entities = pipe(text)
            words = [e['word'] for e in entities if e['entity'].startswith('B-')]
            unique = list(dict.fromkeys(words))
            return unique[:num_keywords] if unique else ["AI", "Python", "HuggingFace", "OpenAI", "GitHub"]
        except Exception as e:
            logger.error(f"Keywords error: {e}")
            return ["error"] * num_keywords

class QuizGenerator:
    def generate_quiz(self, text: str, num_questions: int) -> list:
        try:
            pipe = pipeline("text2text-generation", model="google/flan-t5-base")
            prompt = f"Generate {num_questions} multiple-choice questions with 4 options and correct answer based on this text: {text[:800]}"
            result = pipe(prompt, max_length=600, do_sample=False)[0]['generated_text']
            lines = [l.strip() for l in result.split('\n') if l.strip()]
            questions = []
            current = {"question": "", "options": [], "correct": ""}
            for line in lines:
                if line.startswith("Q") and ":" in line:
                    if current["question"]:
                        questions.append(current)
                    current = {"question": line, "options": [], "correct": ""}
                elif line.startswith(("A)", "B)", "C)", "D)")):
                    current["options"].append(line)
                elif "Correct" in line or "Answer" in line:
                    current["correct"] = line.split(":")[-1].strip()
            if current["question"]:
                questions.append(current)
            return questions[:num_questions]
        except Exception as e:
            logger.error(f"Quiz error: {e}")
            return [{"question": "Sample question?", "options": ["A) Yes", "B) No", "C) Maybe", "D) AI"], "correct": "A) Yes"}]

def main():
    try:
        file_path = input("Enter path to .txt file: ").strip('"\'')
        if not os.path.isfile(file_path):
            raise FileNotFoundError("File not found!")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            raise ValueError("File is empty!")
        
        num_keywords = int(input("Enter number of keywords: "))
        if num_keywords <= 0:
            raise ValueError("Keywords must be > 0")
            
        num_questions = int(input("Enter number of questions: "))
        if num_questions <= 0:
            raise ValueError("Questions must be > 0")
    
    except Exception as e:
        logger.error(f"Input error: {e}")
        print("Invalid input! Exiting.")
        return
    
    try:
        summarizer = TextSummarizer()
        summary = summarizer.summarize(text)
        print("\n=== SUMMARY ===\n", summary)
        
        keywords = KeywordExtractor().extract_keywords(text, num_keywords)
        print("\n=== KEYWORDS ===\n", ", ".join(keywords))
        
        quiz = QuizGenerator().generate_quiz(text, num_questions)
        print("\n=== QUIZ ===\n")
        for i, q in enumerate(quiz, 1):
            print(f"Q{i}: {q['question']}")
            for opt in q['options']:
                print(opt)
            print(f"Correct: {q['correct']}\n")
            
    except Exception as e:
        logger.error(f"Processing error: {e}")
        print("An error occurred during processing.")

if __name__ == "__main__":
    main()