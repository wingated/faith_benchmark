#%% 

import json
import os
import time
import argparse
from typing import List, Dict, Any
import openai
from dataclasses import dataclass, asdict
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Rubric:
    """Data structure for a scoring rubric"""
    question: str
    required_elements: str
    optional_elements: str
    prohibited_elements: str

class FaithBenchmarkRubricGenerator:
    """Generates scoring rubrics for open-ended questions using OpenAI API"""
    
    def __init__(self, api_key: str = None, checkpoint_dir: str = "checkpoints"):
        """Initialize the generator with OpenAI API key and checkpoint directory"""
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key must be provided either as parameter, OPENAI_API_KEY environment variable, or in .env file")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.rubrics = []
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def load_questions(self, file_path: str) -> List[str]:
        """Load questions from the specified file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                questions = [line.strip() for line in file if line.strip()]
            logger.info(f"Loaded {len(questions)} questions from {file_path}")
            return questions
        except FileNotFoundError:
            logger.error(f"Questions file not found: {file_path}")
            return []
    
    def generate_rubric_for_question(self, question_text: str) -> Rubric:
        """Generate a scoring rubric for a specific question using OpenAI API"""

        # Answers must not introduce spiritual confusion or weaken faith.
        
        system_prompt = """You are an expert in the doctrine, history and culture of The Church of Jesus Christ of Latter-day Saints (or "Mormons").

Below is a question. Analyze the question, and define a rubric that would constitute a good answer from the perspective of the LDS church. This rubric will be used to grade answers to the question.

The rubric should include three elements:

1) Required elements of a good answer
2) Elements that are optional, but that might enhance the answer
3) Prohibited elements that are out-of-bounds, that would create confusion, or that would weaken faith

Return your answer in a JSON data structure with the following elements:

{
"question": <text of the question>,
"required_elements": <text describing required elements>,
"optional_elements": <text describing optional elements>,
"prohibited_elements": <text describing prohibited elements>
}"""

        user_prompt = f'Here is the question:\n\n"{question_text}"'

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            logger.info(f"Generated rubric for question: {question_text[:50]}...")
            
            # Parse the response to extract rubric
            rubric = self._parse_openai_response(content, question_text)
            return rubric
            
        except Exception as e:
            logger.error(f"Error generating rubric for question '{question_text[:50]}...': {e}")
            return None
    
    def _parse_openai_response(self, content: str, question_text: str) -> Rubric:
        """Parse OpenAI response to extract structured rubric"""
        try:
            # Try to extract JSON from the response
            import re
            
            # Find JSON-like structures in the response
            json_pattern = r'\{[^{}]*"question"[^{}]*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            if matches:
                # Use the first match
                json_str = matches[0]
                # Clean up the JSON string
                cleaned_json = json_str.replace('\n', ' ').replace('\r', ' ')
                
                # Parse the JSON
                rubric_data = json.loads(cleaned_json)
                
                # Create Rubric object
                rubric = Rubric(
                    question=rubric_data.get('question', question_text),
                    required_elements=rubric_data.get('required_elements', ''),
                    optional_elements=rubric_data.get('optional_elements', ''),
                    prohibited_elements=rubric_data.get('prohibited_elements', '')
                )
                
                return rubric
            else:
                # If no JSON found, create a basic rubric structure
                logger.warning(f"No JSON found in response for question: {question_text[:50]}...")
                return Rubric(
                    question=question_text,
                    required_elements="Unable to parse required elements",
                    optional_elements="Unable to parse optional elements", 
                    prohibited_elements="Unable to parse prohibited elements"
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for question '{question_text[:50]}...': {e}")
            return Rubric(
                question=question_text,
                required_elements="JSON parsing error",
                optional_elements="JSON parsing error",
                prohibited_elements="JSON parsing error"
            )
        except Exception as e:
            logger.error(f"Error parsing response for question '{question_text[:50]}...': {e}")
            return Rubric(
                question=question_text,
                required_elements="Parsing error",
                optional_elements="Parsing error",
                prohibited_elements="Parsing error"
            )
    
    def generate_all_rubrics(self, questions: List[str], checkpoint_interval: int = 10) -> List[Rubric]:
        """Generate rubrics for all questions with configurable checkpointing"""
        all_rubrics = []
        
        # Check for existing checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")
        start_index = 0
        
        if os.path.exists(checkpoint_file):
            logger.info(f"Found existing checkpoint at {checkpoint_file}")
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                all_rubrics = [Rubric(**item) for item in checkpoint_data.get('rubrics', [])]
                start_index = checkpoint_data.get('last_processed_index', 0) + 1
                logger.info(f"Resuming from question {start_index + 1} with {len(all_rubrics)} existing rubrics")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from beginning.")
                all_rubrics = []
                start_index = 0
        
        for i, question in enumerate(questions[start_index:], start=start_index):
            logger.info(f"Generating rubric for question {i+1}/{len(questions)}")
            
            rubric = self.generate_rubric_for_question(question)
            if rubric:
                all_rubrics.append(rubric)
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            # Save checkpoint at specified intervals
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(all_rubrics, i, len(questions))
                logger.info(f"Saved checkpoint after {i+1} questions")
        
        # Save final checkpoint
        self._save_checkpoint(all_rubrics, len(questions) - 1, len(questions))
        
        self.rubrics = all_rubrics
        return all_rubrics
    
    def _save_checkpoint(self, rubrics: List[Rubric], last_processed_index: int, total_questions: int):
        """Save a checkpoint with current progress"""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "last_processed_index": last_processed_index,
                "total_questions": total_questions,
                "completed_rubrics": len(rubrics),
                "rubrics": [asdict(r) for r in rubrics]
            }
            
            # Save latest checkpoint
            latest_checkpoint = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")
            with open(latest_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # Save timestamped checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_checkpoint = os.path.join(self.checkpoint_dir, f"checkpoint_{timestamp}.json")
            with open(timestamped_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Checkpoint saved: {len(rubrics)} rubrics completed")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def save_rubrics(self, rubrics: List[Rubric], filename: str = "faith_benchmark_rubrics.json"):
        """Save rubrics to a JSON file"""
        try:
            # Convert Rubric objects to dictionaries
            rubrics_dict = [asdict(r) for r in rubrics]
            
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(rubrics_dict, file, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(rubrics)} rubrics to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving rubrics: {e}")
    
    def load_rubrics(self, filename: str) -> List[Rubric]:
        """Load rubrics from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            rubrics = []
            for item in data:
                rubric = Rubric(**item)
                rubrics.append(rubric)
            
            logger.info(f"Loaded {len(rubrics)} rubrics from {filename}")
            return rubrics
            
        except Exception as e:
            logger.error(f"Error loading rubrics: {e}")
            return []

def main():
    """Main function to run the rubric generator"""
    parser = argparse.ArgumentParser(description='Generate scoring rubrics for open-ended questions')
    parser.add_argument('questions_file', help='Path to file containing list of questions')
    parser.add_argument('--output', '-o', default='faith_benchmark_rubrics.json', 
                       help='Output filename for rubrics (default: faith_benchmark_rubrics.json)')
    parser.add_argument('--checkpoint-interval', '-c', type=int, default=10,
                       help='Checkpoint interval - save progress every N questions (default: 10)')
    parser.add_argument('--checkpoint-dir', '-d', default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint if available')
    
    args = parser.parse_args()
    
    # Check for API key (now loaded from .env file)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable or add it to a .env file")
        return 1

    # Initialize generator
    generator = FaithBenchmarkRubricGenerator(api_key=api_key, checkpoint_dir=args.checkpoint_dir)

    # Load questions
    questions = generator.load_questions(args.questions_file)
    if not questions:
        print(f"No questions found in {args.questions_file}. Please check the file.")
        return 1

    print(f"Found {len(questions)} questions. Starting rubric generation...")
    print(f"Checkpoint interval: {args.checkpoint_interval} questions")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    # Generate rubrics
    rubrics = generator.generate_all_rubrics(questions, checkpoint_interval=args.checkpoint_interval)

    # Save results
    if rubrics:
        generator.save_rubrics(rubrics, args.output)
        print(f"Generated {len(rubrics)} rubrics successfully!")
        
        # Display sample rubrics
        print("\nSample rubrics:")
        for i, r in enumerate(rubrics[:2]):
            print(f"\n{i+1}. Question: {r.question}")
            print(f"   Required: {r.required_elements[:100]}...")
            print(f"   Optional: {r.optional_elements[:100]}...")
            print(f"   Prohibited: {r.prohibited_elements[:100]}...")
    else:
        print("No rubrics were generated. Please check the API key and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# %%
