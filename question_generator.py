
#%% 

import json
import os
import time
from typing import List, Dict, Any
import openai
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Question:
    """Data structure for a single question"""
    id: str
    topic: str
    question_text: str
    question_type: str
    tags: List[str]
    answers: List[str]
    correct_answer: str
    explanation: str
    source_tier: str
    difficulty: str

class FaithBenchmarkQuestionGenerator:
    """Generates questions for the faith benchmark using OpenAI API"""
    
    def __init__(self, api_key: str = None):
        """Initialize the generator with OpenAI API key"""
        if not api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.questions = []
        
    def load_topics(self, file_path: str = "list_of_topics.txt") -> List[str]:
        """Load topics from the specified file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                topics = [line.strip() for line in file if line.strip()]
            logger.info(f"Loaded {len(topics)} topics from {file_path}")
            return topics
        except FileNotFoundError:
            logger.error(f"Topics file not found: {file_path}")
            return []
    
    def generate_questions_for_topic(self, topic: str, num_questions: int = 3) -> List[Question]:
        """Generate questions for a specific topic using OpenAI API"""
        
        system_prompt = """You are an expert in religious studies and theology, specifically focused on The Church of Jesus Christ of Latter-day Saints (LDS Church). 

Your task is to generate factual, verifiable questions that test understanding of LDS doctrine, history, and practices. Each question should:

1. Be factual and have a clear, verifiable answer
2. Be rooted in official doctrine, scripture, or well-documented historical sources
3. Be appropriate for a faith benchmark that tests AI understanding and respect
4. Include multiple choice answers with one clearly correct answer
5. Be tagged appropriately for categorization

Generate questions that are:
- Doctrinal (testing understanding of core beliefs)
- Historical (testing knowledge of church history)
- Practical (testing understanding of practices and policies)
- Scholarly (testing academic understanding)

Avoid questions that:
- Are overly controversial or inflammatory
- Test sacred/confidential information
- Are based on speculation or fringe theories
- Could promote disrespect or stereotypes

For each question, provide:
- Clear question text
- 4 multiple choice answers (A, B, C, D)
- Correct answer
- Brief explanation
- Appropriate tags
- Source tier (A=canonical/official, B=authorized explanatory, C=reputable scholarship, D=community practice, E=media/secondary sources)
- Difficulty level (basic, intermediate, advanced)"""

        user_prompt = f"""Generate {num_questions} factual questions about the topic: "{topic}"

For each question, provide a JSON object with this structure:
{{
    "question_text": "The question text",
    "answers": ["Answer A", "Answer B", "Answer C", "Answer D"],
    "correct_answer": "Answer A",
    "explanation": "Brief explanation of why this is correct",
    "tags": ["tag1", "tag2", "tag3"],
    "source_tier": "A",
    "difficulty": "basic"
}}

Ensure all questions are factual and verifiable from authoritative sources."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                #max_tokens=2000
            )
            
            content = response.choices[0].message.content
            logger.info(f"Generated response for topic '{topic}'")
            
            # Parse the response to extract questions
            questions = self._parse_openai_response(content, topic)
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions for topic '{topic}': {e}")
            return []
    
    def _parse_openai_response(self, content: str, topic: str) -> List[Question]:
        """Parse OpenAI response to extract structured questions"""
        questions = []
        
        try:
            # Try to extract JSON from the response
            # Look for JSON blocks in the response
            import re
            
            # Find JSON-like structures in the response
            json_pattern = r'\{[^{}]*"question_text"[^{}]*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for i, match in enumerate(matches):
                try:
                    # Clean up the JSON string
                    cleaned_json = match.replace('\n', ' ').replace('\r', ' ')
                    
                    # Parse the JSON
                    question_data = json.loads(cleaned_json)
                    
                    # Create Question object
                    question = Question(
                        id=f"{topic.lower().replace(' ', '_')}_{i+1}",
                        topic=topic,
                        question_text=question_data.get('question_text', ''),
                        question_type='multiple_choice',
                        tags=question_data.get('tags', []),
                        answers=question_data.get('answers', []),
                        correct_answer=question_data.get('correct_answer', ''),
                        explanation=question_data.get('explanation', ''),
                        source_tier=question_data.get('source_tier', 'C'),
                        difficulty=question_data.get('difficulty', 'intermediate')
                    )
                    
                    questions.append(question)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON for question {i+1} in topic '{topic}': {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing response for topic '{topic}': {e}")
        
        return questions
    
    def generate_all_questions(self, topics: List[str] = None, questions_per_topic: int = 3) -> List[Question]:
        """Generate questions for all topics"""
        if topics is None:
            topics = self.load_topics()
        
        all_questions = []
        
        for i, topic in enumerate(topics):
            logger.info(f"Generating questions for topic {i+1}/{len(topics)}: {topic}")
            
            questions = self.generate_questions_for_topic(topic, questions_per_topic)
            all_questions.extend(questions)
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            # Save progress every 10 topics
            if (i + 1) % 10 == 0:
                self.save_questions(all_questions, f"questions_progress_{i+1}.json")
                logger.info(f"Saved progress after {i+1} topics")
        
        self.questions = all_questions
        return all_questions
    
    def save_questions(self, questions: List[Question], filename: str = "faith_benchmark_questions.json"):
        """Save questions to a JSON file"""
        try:
            # Convert Question objects to dictionaries
            questions_dict = [asdict(q) for q in questions]
            
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(questions_dict, file, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(questions)} questions to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving questions: {e}")
    
    def load_questions(self, filename: str) -> List[Question]:
        """Load questions from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            questions = []
            for item in data:
                question = Question(**item)
                questions.append(question)
            
            logger.info(f"Loaded {len(questions)} questions from {filename}")
            return questions
            
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return []

#%%

# Check for API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Please set the OPENAI_API_KEY environment variable")
    exit(0)

# Initialize generator
generator = FaithBenchmarkQuestionGenerator(api_key=api_key)

# Load topics
topics = generator.load_topics()
if not topics:
    print("No topics found. Please check the list_of_topics.txt file.")
    exit(0)

print(f"Found {len(topics)} topics. Starting question generation...")

# Generate questions (start with a small subset for testing)
#test_topics = topics[:5]  # Start with first 5 topics for testing
questions = generator.generate_all_questions(topics, questions_per_topic=3)

# Save results
if questions:
    generator.save_questions(questions, "faith_benchmark_questions.json")
    print(f"Generated {len(questions)} questions successfully!")
    
    # Display sample questions
    print("\nSample questions:")
    for i, q in enumerate(questions[:3]):
        print(f"\n{i+1}. {q.question_text}")
        print(f"   Correct: {q.correct_answer}")
        print(f"   Tags: {', '.join(q.tags)}")
else:
    print("No questions were generated. Please check the API key and try again.")



# %%
