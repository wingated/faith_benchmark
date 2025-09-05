#!/usr/bin/env python3
"""
Faith Benchmark Answer Generator Script
Generates answers to open-ended questions using OpenRouter API
"""

import json
import os
import sys
import argparse
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

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
class AnswerResult:
    """Data structure for a single answer result"""
    question: str
    model_name: str
    answer: str
    timestamp: str
    response_time: float
    raw_response: str

class AnswerGenerator:
    """Generates answers to open-ended questions using OpenRouter API"""

    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the answer generator"""
        self.model_config = model_config
        self.client = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the OpenRouter client for unified API access"""
        try:
            import openai

            # Use OpenRouter API with OpenAI client
            self.client = openai.OpenAI(
                api_key=self.model_config.get("api_key"),
                base_url="https://openrouter.ai/api/v1"
            )

            logger.info(f"Initialized OpenRouter client for model {self.model_config['model_name']} successfully")

        except ImportError as e:
            logger.error(f"Failed to import OpenAI package: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise

    def generate_answer(self, question: str) -> AnswerResult:
        """Generate an answer for a single question and return the result"""

        start_time = time.time()

        # Construct the prompt
        prompt = self._construct_prompt(question)

        try:
            # Get response from model
            response = self._get_model_response(prompt)

            # Calculate response time
            response_time = time.time() - start_time

            # Create answer result
            result = AnswerResult(
                question=question,
                model_name=self.model_config.get("model_name"),
                answer=response,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                response_time=response_time,
                raw_response=response
            )

            return result

        except Exception as e:
            logger.error(f"Error generating answer for question: {e}")
            # Return error result
            return AnswerResult(
                question=question,
                model_name=self.model_config.get("model_name"),
                answer=f"ERROR: {str(e)}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                response_time=time.time() - start_time,
                raw_response=f"Error: {str(e)}"
            )

    def _construct_prompt(self, question: str) -> str:
        """Construct the prompt for the model"""

        prompt = f"""You are an expert in the doctrine, history and culture of The Church of Jesus Christ of Latter-day Saints (or "Mormons").

Please provide a thoughtful, accurate, and faithful answer to the following question. Your answer should be based on official LDS teachings and doctrine.

Question: {question}

Please provide a comprehensive answer that demonstrates understanding of LDS doctrine and teachings. Be respectful, accurate, and helpful in your response."""

        return prompt

    def _get_model_response(self, prompt: str) -> str:
        """Get response from the model via OpenRouter"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config.get("model_name"),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.model_config.get("temperature", 0.3),
                max_tokens=self.model_config.get("max_tokens", 1000),
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error getting response from {self.model_config.get('model_name')}: {e}")
            raise

class BenchmarkAnswerGenerator:
    """Main class for generating answers to benchmark questions"""

    def __init__(self, model_config: Dict[str, Any], rubrics_file: str, output_dir: str = "results"):
        """Initialize the answer generator"""
        self.model_config = model_config
        self.rubrics_file = rubrics_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load rubrics
        self.rubrics = self._load_rubrics()
        logger.info(f"Loaded {len(self.rubrics)} questions from {rubrics_file}")

        # Initialize results storage
        self.results: List[AnswerResult] = []
        self.completed_questions: set = set()

        # Load checkpoint if exists
        self._load_checkpoint()

    def _load_rubrics(self) -> List[Dict[str, Any]]:
        """Load rubrics from JSON file"""
        try:
            with open(self.rubrics_file, 'r', encoding='utf-8') as file:
                rubrics = json.load(file)
            return rubrics
        except Exception as e:
            logger.error(f"Failed to load rubrics from {self.rubrics_file}: {e}")
            raise

    def _get_checkpoint_file(self) -> Path:
        """Get checkpoint file path for a specific model"""
        # Replace slashes with underscores to avoid file path issues
        safe_rubric = self.rubrics_file.replace("/", "_")
        safe_model = self.model_config["model_name"].replace("/", "_")
        return self.output_dir / f"answer_checkpoint_{safe_rubric}_{safe_model}.json"

    def _save_checkpoint(self):
        """Save checkpoint for a specific model"""
        checkpoint_file = self._get_checkpoint_file()

        checkpoint_data = {
            "model_name": self.model_config["model_name"],
            "rubrics_file": self.rubrics_file,
            "completed_questions": list(self.completed_questions),
            "results": [asdict(result) for result in self.results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as file:
                json.dump(checkpoint_data, file, indent=2)
            logger.info(f"Saved checkpoint for {self.model_config['model_name']}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        # This will be called after model initialization
        pass

    def _load_model_checkpoint(self):
        """Load checkpoint for a specific model"""
        checkpoint_file = self._get_checkpoint_file()

        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as file:
                    checkpoint_data = json.load(file)

                # Restore completed questions
                self.completed_questions = set(checkpoint_data.get("completed_questions", []))

                # Restore results
                self.results = []
                for result_data in checkpoint_data.get("results", []):
                    result = AnswerResult(**result_data)
                    self.results.append(result)

                logger.info(f"Loaded checkpoint for {self.model_config['model_name']}: {len(self.completed_questions)} completed questions, {len(self.results)} results")

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                # Continue without checkpoint
        else:
            logger.info(f"No checkpoint found for {self.model_config['model_name']}, starting fresh")

    def generate_answers(self, checkpoint_interval: int = 10):
        """Generate answers for all questions"""

        logger.info(f"Starting answer generation for {self.model_config['model_name']}")

        # Load checkpoint for this model
        self._load_model_checkpoint()

        # Initialize answer generator
        try:
            generator = AnswerGenerator(self.model_config)
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_config['model_name']}: {e}")
            return

        # Get questions that haven't been completed
        remaining_questions = [r for r in self.rubrics if r["question"] not in self.completed_questions]

        if not remaining_questions:
            logger.info(f"All questions already completed for {self.model_config['model_name']}")
            return

        logger.info(f"Generating answers for {len(remaining_questions)} remaining questions for {self.model_config['model_name']}")

        # Get delay from environment or use default
        delay = float(os.getenv("DELAY_BETWEEN_QUESTIONS", "1.0"))

        # Generate answer for each question
        for i, rubric in enumerate(remaining_questions):
            question = rubric["question"]
            logger.info(f"{self.model_config['model_name']} - Generating answer {i+1}/{len(remaining_questions)}: {question[:50]}...")

            try:
                # Generate the answer
                result = generator.generate_answer(question)

                # Store result
                self.results.append(result)
                self.completed_questions.add(question)

                # Log result
                logger.info(f"{self.model_config['model_name']} - Generated answer for: {question[:50]}...")

                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint()
                    logger.info(f"Checkpoint saved after {i+1} questions")

                # Add delay to avoid rate limiting
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Error generating answer for question: {e}")
                continue

        # Final checkpoint
        self._save_checkpoint()
        logger.info(f"Completed answer generation for {self.model_config['model_name']}")

    def export_results(self):
        """Export results to JSON"""
        self._export_json()

    def _export_json(self):
        """Export results to JSON"""
        # Replace slashes with underscores to avoid file path issues
        safe_rubric = self.rubrics_file.replace("/", "_")
        safe_model = self.model_config["model_name"].replace("/", "_")
        json_file = self.output_dir / f"answer_{safe_rubric}_{safe_model}.json"

        try:
            # Convert results to list of dictionaries
            results_data = [asdict(result) for result in self.results]

            with open(json_file, 'w', encoding='utf-8') as file:
                json.dump(results_data, file, indent=2, ensure_ascii=False)

            logger.info(f"Exported results to {json_file}")

            # Print summary
            total_questions = len(self.results)
            successful_answers = sum(1 for r in self.results if not r.answer.startswith("ERROR"))

            print(f"\n📊 Results Summary for {self.model_config['model_name']}:")
            print(f"Total Questions: {total_questions}")
            print(f"Successful Answers: {successful_answers}")
            print(f"Failed Answers: {total_questions - successful_answers}")
            print(f"Results saved to: {json_file}")

        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model using OpenRouter"""

    # Get OpenRouter API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found. Please set this environment variable.")

    temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "1000"))

    return {
        "api_key": api_key,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

def load_environment_config() -> Dict[str, Any]:
    """Load configuration from environment variables and .env file"""

    config = {}

    # OpenRouter API Key
    config["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY")

    # Model configurations
    config["openrouter_temperature"] = float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
    config["openrouter_max_tokens"] = int(os.getenv("OPENROUTER_MAX_TOKENS", "1000"))

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

    # Check OpenRouter API key
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    status = "✅" if openrouter_key else "❌"
    print(f"{status} OpenRouter: {'Configured' if openrouter_key else 'Not configured'}")

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
    print("The Faith Benchmark Answer Generator script can be configured using:")
    print("1. Environment variables")
    print("2. A .env file in the current directory")
    print("3. Command line arguments (which override environment variables)")

    print("\n📁 .env File Setup:")
    print("1. Copy env.example to .env:")
    print("   cp env.example .env")
    print("2. Edit .env and add your OpenRouter API key:")
    print("   OPENROUTER_API_KEY=your_actual_key_here")

    print("\n🔑 Required API Keys:")
    print("- OPENROUTER_API_KEY: For accessing models via OpenRouter")
    print("  Get your key from: https://openrouter.ai/keys")

    print("\n⚙️  Optional Configuration:")
    print("- OPENROUTER_TEMPERATURE: Temperature setting (default: 0.3)")
    print("- OPENROUTER_MAX_TOKENS: Max tokens per response (default: 1000)")
    print("- CHECKPOINT_INTERVAL: Save progress every N questions (default: 10)")
    print("- OUTPUT_DIRECTORY: Directory for results (default: 'results')")
    print("- DELAY_BETWEEN_QUESTIONS: Delay in seconds (default: 1.0)")
    print("- LOG_LEVEL: Logging level (default: INFO)")

    print("\n�� Usage Examples:")
    print("1. Check configuration status:")
    print("   python generate_answers.py --status")
    print("2. Generate answers with GPT-4 via OpenRouter:")
    print("   python generate_answers.py --model openai/gpt-4 --rubrics nursery_rubric.json")
    print("3. Generate answers with Claude via OpenRouter:")
    print("   python generate_answers.py --model anthropic/claude-3-sonnet --rubrics nursery_rubric.json")
    print("4. Override environment settings:")
    print("   python generate_answers.py --model openai/gpt-4 --checkpoint-interval 5 --rubrics nursery_rubric.json")

    print("=" * 50)

def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Generate answers to open-ended faith benchmark questions")
    parser.add_argument("--model",
                       help="Model to use (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet', 'google/gemini-pro')")
    parser.add_argument("--rubrics", default="nursery_rubric.json",
                       help="Path to rubrics JSON file")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results and checkpoints (overrides OUTPUT_DIRECTORY env var)")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                       help="Save checkpoint every N questions (overrides CHECKPOINT_INTERVAL env var)")
    parser.add_argument("--status", action="store_true",
                       help="Show environment configuration status and exit")
    parser.add_argument("--help-env", action="store_true",
                       help="Show environment configuration help and exit")

    args = parser.parse_args()

    # Check if model is required (not needed for help commands)
    if not args.status and not args.help_env and not args.model:
        parser.error("--model is required unless using --status or --help-env")

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

    # Check if rubrics file exists
    if not os.path.exists(args.rubrics):
        print(f"❌ Rubrics file not found: {args.rubrics}")
        print("Please specify the correct path to the rubrics JSON file.")
        sys.exit(1)

    # Get model configuration
    try:
        model_config = get_model_config(args.model)
    except ValueError as e:
        print(f"❌ {e}")
        print("\n💡 Environment Configuration Help:")
        print("   Set OpenRouter API key in environment variables or .env file:")
        print("   - OPENROUTER_API_KEY for accessing models via OpenRouter")
        print("\n   Run with --help-env for detailed configuration help")
        print("   Run with --status to see current configuration")
        sys.exit(1)

    # Initialize answer generator
    try:
        generator = BenchmarkAnswerGenerator(model_config, args.rubrics, output_dir)
    except Exception as e:
        print(f"❌ Failed to initialize answer generator: {e}")
        sys.exit(1)

    # Generate answers
    print(f"🚀 Starting answer generation for {args.model}")
    print(f"📁 Rubrics file: {args.rubrics}")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Checkpoint interval: {checkpoint_interval} questions")
    print(f"⏱️  Delay between questions: {env_config['delay_between_questions']} seconds")
    print("=" * 60)

    try:
        generator.generate_answers(checkpoint_interval)

        # Export results
        generator.export_results()

        print(f"\n✅ Answer generation completed for {args.model}!")

    except KeyboardInterrupt:
        print(f"\n⚠️  Answer generation interrupted. Progress has been saved to checkpoint.")
        print("You can resume by running the same command again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Answer generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
