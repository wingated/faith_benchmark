#!/usr/bin/env python3
"""
Faith Benchmark Test Accuracy Script
Tests different language models on benchmark questions and exports results to CSV
"""

import json
import csv
import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import re

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    print("❌ dotenv not available, continuing without it")
    pass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Data structure for a single test result"""
    question_id: str
    topic: str
    question_text: str
    tags: List[str]
    correct_answer: str
    model_answer: str
    is_correct: bool
    model_name: str
    timestamp: str
    response_time: float
    raw_response: str

class ModelTester:
    """Tests language models on benchmark questions"""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """Initialize the model tester"""
        self.model_name = model_name
        self.model_config = model_config
        self.client = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specific model client"""
        try:
            if self.model_name.lower() == "openai":
                import openai
                self.client = openai.OpenAI(api_key=self.model_config.get("api_key"))
                
            elif self.model_name.lower() == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.model_config.get("api_key"))
                self.client = genai.GenerativeModel(self.model_config.get("model", "gemini-pro"))

            elif self.model_name.lower() == "claude":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.model_config.get("api_key"))
                
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            self.model = self.model_config.get("model")

            logger.info(f"Initialized {self.model_name} model successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required package for {self.model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name} model: {e}")
            raise
    
    def test_question(self, question: Dict[str, Any]) -> TestResult:
        """Test a single question and return the result"""
        
        start_time = time.time()
        
        # Construct the prompt
        prompt = self._construct_prompt(question)
        
        try:
            # Get response from model
            response = self._get_model_response(prompt)
            
            # Extract the answer (single letter)
            model_answer = self._extract_answer(response)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Create test result
            result = TestResult(
                question_id=question["id"],
                topic=question["topic"],
                question_text=question["question_text"],
                tags=question["tags"],
                correct_answer=question["correct_answer"],
                model_answer=model_answer,
                is_correct=model_answer == question["correct_answer"],
                model_name=self.model_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                response_time=response_time,
                raw_response=response
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing question {question['id']}: {e}")
            # Return error result
            return TestResult(
                question_id=question["id"],
                topic=question["topic"],
                question_text=question["question_text"],
                tags=question["tags"],
                correct_answer=question["correct_answer"],
                model_answer="ERROR",
                is_correct=False,
                model_name=self.model_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                response_time=time.time() - start_time,
                raw_response=f"Error: {str(e)}"
            )
    
    def _construct_prompt(self, question: Dict[str, Any]) -> str:
        """Construct the prompt for the model"""
        
        prompt = f"""You are taking a multiple choice test about The Church of Jesus Christ of Latter-day Saints.

Question: {question['question_text']}

Options:
A) {question['answers'][0]}
B) {question['answers'][1]}
C) {question['answers'][2]}
D) {question['answers'][3]}

Instructions: Answer with ONLY a single letter (A, B, C, or D). Do not include any other text, explanation, or formatting.

Answer:"""
        
        return prompt
    
    def _get_model_response(self, prompt: str) -> str:
        """Get response from the specific model"""
        
        if self.model_name.lower() == "openai":
            if self.model == "gpt-5":
                response = self.client.chat.completions.create(
                  model=self.model,
                  messages=[{"role": "user", "content": prompt}],
                  temperature=1.0,  # this is the only supported value
                  max_completion_tokens=self.model_config.get("max_tokens", 10),
                )
            else:
                response = self.client.chat.completions.create(
                  model=self.model,
                  messages=[{"role": "user", "content": prompt}],
                  temperature=self.model_config.get("temperature", 0.0),
                  max_tokens=self.model_config.get("max_tokens", 10)
                )
            return response.choices[0].message.content.strip()
            
        elif self.model_name.lower() == "gemini":
            response = self.client.generate_content(prompt)
            return response.text.strip()
            
        elif self.model_name.lower() == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.model_config.get("max_tokens", 10),
                temperature=self.model_config.get("temperature", 0.0),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _extract_answer(self, response: str) -> str:
        """Extract the single letter answer from the model response"""
        
        # Clean the response
        response = response.strip().upper()
        
        # Look for single letter answers
        if len(response) == 1 and response in ['A', 'B', 'C', 'D']:
            return response
        
        # Look for letter followed by punctuation or space
        match = re.search(r'^([ABCD])[\.\)\s]?', response)
        if match:
            return match.group(1)
        
        # Look for letter anywhere in the response
        match = re.search(r'[ABCD]', response)
        if match:
            return match.group(0)
        
        # If no clear answer found, return the first character or 'X'
        if response and response[0] in ['A', 'B', 'C', 'D']:
            return response[0]
        
        return 'X'  # Invalid answer

class BenchmarkTester:
    """Main class for running benchmark tests"""
    
    def __init__(self, questions_file: str, output_dir: str = "results"):
        """Initialize the benchmark tester"""
        self.questions_file = questions_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load questions
        self.questions = self._load_questions()
        logger.info(f"Loaded {len(self.questions)} questions from {questions_file}")
        
        # Initialize results storage
        self.results: List[TestResult] = []
        self.completed_questions: set = set()
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from JSON file"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as file:
                questions = json.load(file)
            return questions
        except Exception as e:
            logger.error(f"Failed to load questions from {self.questions_file}: {e}")
            raise
    
    def _get_checkpoint_file(self, model_name: str, model: str) -> Path:
        """Get checkpoint file path for a specific model"""
        return self.output_dir / f"checkpoint_{model_name}_{model}.json"
    
    def _save_checkpoint(self, model_name: str, model: str):
        """Save checkpoint for a specific model"""
        checkpoint_file = self._get_checkpoint_file(model_name, model)
        
        checkpoint_data = {
            "model_name": model_name,
            "model": model,
            "completed_questions": list(self.completed_questions),
            "results": [asdict(result) for result in self.results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as file:
                json.dump(checkpoint_data, file, indent=2)
            logger.info(f"Saved checkpoint for {model_name}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        # This will be called after model initialization
        pass
    
    def _load_model_checkpoint(self, model_name: str, model: str):
        """Load checkpoint for a specific model"""
        checkpoint_file = self._get_checkpoint_file(model_name, model)
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as file:
                    checkpoint_data = json.load(file)
                
                # Restore completed questions
                self.completed_questions = set(checkpoint_data.get("completed_questions", []))
                
                # Restore results
                self.results = []
                for result_data in checkpoint_data.get("results", []):
                    result = TestResult(**result_data)
                    self.results.append(result)
                
                logger.info(f"Loaded checkpoint for {model_name} / {model}: {len(self.completed_questions)} completed questions, {len(self.results)} results")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                # Continue without checkpoint
        else:
            logger.info(f"No checkpoint found for {model_name}, starting fresh")
    
    def run_tests(self, model_name: str, model_config: Dict[str, Any], checkpoint_interval: int = 10):
        """Run tests for a specific model"""
        
        logger.info(f"Starting tests for {model_name}")
        
        # Load checkpoint for this model
        self._load_model_checkpoint(model_name, model_config["model"])
        
        # Initialize model tester
        try:
            tester = ModelTester(model_name, model_config)
        except Exception as e:
            logger.error(f"Failed to initialize {model_name}: {e}")
            return
        
        # Get questions that haven't been completed
        remaining_questions = [q for q in self.questions if q["id"] not in self.completed_questions]
        
        if not remaining_questions:
            logger.info(f"All questions already completed for {model_name}")
            return
        
        logger.info(f"Testing {len(remaining_questions)} remaining questions for {model_name}")
        
        # Get delay from environment or use default
        delay = float(os.getenv("DELAY_BETWEEN_QUESTIONS", "1.0"))
        
        # Test each question
        for i, question in enumerate(remaining_questions):
            logger.info(f"Testing question {i+1}/{len(remaining_questions)}: {question['id']}")
            
            try:
                # Test the question
                result = tester.test_question(question)
                
                # Store result
                self.results.append(result)
                self.completed_questions.add(question["id"])
                
                # Log result
                status = "✓" if result.is_correct else "✗"
                logger.info(f"{status} {question['id']}: Expected {result.correct_answer}, Got {result.model_answer}")
                
                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(model_name, model_config["model"])
                    logger.info(f"Checkpoint saved after {i+1} questions")
                
                # Add delay to avoid rate limiting
                if delay > 0:
                    time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error testing question {question['id']}: {e}")
                continue
        
        # Final checkpoint
        self._save_checkpoint(model_name, model_config["model"])
        logger.info(f"Completed testing for {model_name} / {model_config['model']}")
    
    def export_results(self, model_name: str, format: str = "csv"):
        """Export results to specified format"""
        
        if format.lower() == "csv":
            self._export_csv(model_name)
        else:
            logger.error(f"Unsupported format: {format}")
    
    def _export_csv(self, model_name: str):
        """Export results to CSV"""
        
        csv_file = self.output_dir / f"results_{model_name}.csv"
        
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Write header
                writer.writerow([
                    "question_id", "topic", "question_text", "tags", 
                    "correct_answer", "model_answer", "is_correct", 
                    "model_name", "timestamp", "response_time", "raw_response"
                ])
                
                # Write results
                for result in self.results:
                    writer.writerow([
                        result.question_id, result.topic, result.question_text,
                        ";".join(result.tags), result.correct_answer, result.model_answer,
                        result.is_correct, result.model_name, result.timestamp,
                        result.response_time, result.raw_response
                    ])
            
            logger.info(f"Exported results to {csv_file}")
            
            # Print summary
            total_questions = len(self.results)
            correct_answers = sum(1 for r in self.results if r.is_correct)
            accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
            
            print(f"\n📊 Results Summary for {model_name}:")
            print(f"Total Questions: {total_questions}")
            print(f"Correct Answers: {correct_answers}")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Results saved to: {csv_file}")
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model from environment variables"""
    
    # Define default configurations
    configs = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "10")),
        },
        "gemini": {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "model": os.getenv("GOOGLE_MODEL"),
            "temperature": float(os.getenv("GOOGLE_TEMPERATURE", "0.0")),
            "max_tokens": int(os.getenv("GOOGLE_MAX_TOKENS", "10")),
        },
        "claude": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": os.getenv("ANTHROPIC_MODEL"),
            "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.0")),
            "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "10")),
        }
    }
    
    if model_name.lower() not in configs:
        raise ValueError(f"Unsupported model: {model_name}")
    
    config = configs[model_name.lower()]
    
    if not config["api_key"]:
        raise ValueError(f"API key not found for {model_name}. Please set the appropriate environment variable.")
    
    return config

def load_environment_config() -> Dict[str, Any]:
    """Load configuration from environment variables and .env file"""
    
    config = {}
    
    # API Keys
    config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    config["google_api_key"] = os.getenv("GOOGLE_API_KEY")
    config["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
    
    # Model configurations
    config["openai_model"] = os.getenv("OPENAI_MODEL")
    config["google_model"] = os.getenv("GOOGLE_MODEL")
    config["anthropic_model"] = os.getenv("ANTHROPIC_MODEL")
    
    # Testing configurations
    config["checkpoint_interval"] = int(os.getenv("CHECKPOINT_INTERVAL", "10"))
    config["output_directory"] = os.getenv("OUTPUT_DIRECTORY", "results")
    config["delay_between_questions"] = float(os.getenv("DELAY_BETWEEN_QUESTIONS", "1.0"))
    
    # Logging configuration
    config["log_level"] = os.getenv("LOG_LEVEL", "INFO")
    
    return config

def print_environment_status():
    """Print the status of environment variables and configuration"""
    
    print("🔧 Environment Configuration Status")
    print("=" * 50)
    
    # Check API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Google Gemini": os.getenv("GOOGLE_API_KEY"),
        "Anthropic Claude": os.getenv("ANTHROPIC_API_KEY")
    }
    
    for provider, key in api_keys.items():
        status = "✅" if key else "❌"
        print(f"{status} {provider}: {'Configured' if key else 'Not configured'}")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file found: {env_file}")
    else:
        print("⚠️  .env file not found (using system environment variables)")
    
    # Show configuration values
    config = load_environment_config()
    print(f"\n📋 Configuration Values:")
    print(f"   Checkpoint Interval: {config['checkpoint_interval']} questions")
    print(f"   Output Directory: {config['output_directory']}")
    print(f"   Delay Between Questions: {config['delay_between_questions']} seconds")
    print(f"   Log Level: {config['log_level']}")
    
    print("=" * 50)

def print_environment_help():
    """Print help information for environment configuration"""
    
    print("💡 Environment Configuration Help")
    print("=" * 50)
    print("The Faith Benchmark Test Accuracy script can be configured using:")
    print("1. Environment variables")
    print("2. A .env file in the current directory")
    print("3. Command line arguments (which override environment variables)")
    
    print("\n📁 .env File Setup:")
    print("1. Copy env.example to .env:")
    print("   cp env.example .env")
    print("2. Edit .env and add your API keys:")
    print("   OPENAI_API_KEY=your_actual_key_here")
    print("   GOOGLE_API_KEY=your_actual_key_here")
    print("   ANTHROPIC_API_KEY=your_actual_key_here")
    
    print("\n🔑 Required API Keys:")
    print("- OPENAI_API_KEY: For testing OpenAI models (GPT-4, GPT-3.5)")
    print("- GOOGLE_API_KEY: For testing Google Gemini models")
    print("- ANTHROPIC_API_KEY: For testing Anthropic Claude models")
    
    print("\n⚙️  Optional Configuration:")
    print("- CHECKPOINT_INTERVAL: Save progress every N questions (default: 10)")
    print("- OUTPUT_DIRECTORY: Directory for results (default: 'results')")
    print("- DELAY_BETWEEN_QUESTIONS: Delay in seconds (default: 1.0)")
    print("- LOG_LEVEL: Logging level (default: INFO)")
    
    print("\n📖 Usage Examples:")
    print("1. Check configuration status:")
    print("   python test_accuracy.py --status")
    print("2. Test with OpenAI (using .env config):")
    print("   python test_accuracy.py --model openai")
    print("3. Override environment settings:")
    print("   python test_accuracy.py --model openai --checkpoint-interval 5")
    
    print("=" * 50)

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Test language models on faith benchmark questions")
    parser.add_argument("--model", required=True, choices=["openai", "gemini", "claude"], 
                       help="Language model to test")
    parser.add_argument("--questions", default="faith_benchmark_questions.json",
                       help="Path to questions JSON file")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results and checkpoints (overrides OUTPUT_DIRECTORY env var)")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                       help="Save checkpoint every N questions (overrides CHECKPOINT_INTERVAL env var)")
    parser.add_argument("--format", default="csv", choices=["csv"],
                       help="Output format for results")
    parser.add_argument("--status", action="store_true",
                       help="Show environment configuration status and exit")
    parser.add_argument("--help-env", action="store_true",
                       help="Show environment configuration help and exit")
    
    args = parser.parse_args()
    
    # Show environment status if requested
    if args.status:
        print_environment_status()
        return
    
    # Show environment help if requested
    if args.help_env:
        print_environment_help()
        return
    
    # Load environment configuration
    env_config = load_environment_config()
    
    # Set logging level from environment
    logging.getLogger().setLevel(getattr(logging, env_config["log_level"]))
    
    # Use environment values with command line overrides
    output_dir = args.output_dir or env_config["output_directory"]
    checkpoint_interval = args.checkpoint_interval or env_config["checkpoint_interval"]
    
    # Check if questions file exists
    if not os.path.exists(args.questions):
        print(f"❌ Questions file not found: {args.questions}")
        print("Please run the question generator first or specify the correct path.")
        sys.exit(1)
    
    # Get model configuration
    try:
        model_config = get_model_config(args.model)
    except ValueError as e:
        print(f"❌ {e}")
        print("\n💡 Environment Configuration Help:")
        print("   Set API keys in environment variables or .env file:")
        print("   - OPENAI_API_KEY for OpenAI models")
        print("   - GOOGLE_API_KEY for Gemini models")
        print("   - ANTHROPIC_API_KEY for Claude models")
        print("\n   Run with --help-env for detailed configuration help")
        print("   Run with --status to see current configuration")
        sys.exit(1)
    
    # Initialize benchmark tester
    try:
        tester = BenchmarkTester(args.questions, output_dir)
    except Exception as e:
        print(f"❌ Failed to initialize benchmark tester: {e}")
        sys.exit(1)
    
    # Run tests
    print(f"🚀 Starting benchmark tests for {args.model}")
    print(f"📁 Questions file: {args.questions}")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Checkpoint interval: {checkpoint_interval} questions")
    print(f"⏱️  Delay between questions: {env_config['delay_between_questions']} seconds")
    print("=" * 60)
    
    try:
        tester.run_tests(args.model, model_config, checkpoint_interval)
        
        # Export results
        tester.export_results(args.model, args.format)
        
        print(f"\n✅ Benchmark testing completed for {args.model}!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Testing interrupted. Progress has been saved to checkpoint.")
        print("You can resume by running the same command again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
