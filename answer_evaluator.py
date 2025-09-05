#!/usr/bin/env python3
"""
Faith Benchmark Answer Evaluator Script
Evaluates answers to open-ended questions using LLM evaluators via OpenRouter API
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
class EvaluationResult:
    """Data structure for a single evaluation result"""
    question: str
    answer: str
    model_name: str
    evaluator_model: str
    analysis: str
    required_elements_rating: int
    prohibited_elements_rating: int
    optional_elements_rating: int
    timestamp: str
    response_time: float
    raw_response: str

class AnswerEvaluator:
    """Evaluates answers using LLM evaluators via OpenRouter API"""

    def __init__(self, evaluator_model_name: str, model_config: Dict[str, Any]):
        """Initialize the answer evaluator"""
        self.evaluator_model_name = evaluator_model_name
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

            self.model = self.model_config.get("model")

            logger.info(f"Initialized OpenRouter client for evaluator model {self.model} successfully")

        except ImportError as e:
            logger.error(f"Failed to import OpenAI package: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise

    def evaluate_answer(self, question: str, answer: str, rubric: Dict[str, str]) -> EvaluationResult:
        """Evaluate an answer against a rubric and return the result"""

        start_time = time.time()

        # Construct the evaluation prompt
        prompt = self._construct_evaluation_prompt(question, answer, rubric)

        try:
            # Get response from evaluator model
            response = self._get_model_response(prompt)

            # Calculate response time
            response_time = time.time() - start_time

            # Parse the evaluation response
            evaluation_data = self._parse_evaluation_response(response)

            # Create evaluation result
            result = EvaluationResult(
                question=question,
                answer=answer,
                model_name="",  # Will be set by caller
                evaluator_model=self.model,
                analysis=evaluation_data.get("analysis", ""),
                required_elements_rating=evaluation_data.get("required_elements_rating", 0),
                prohibited_elements_rating=evaluation_data.get("prohibited_elements_rating", 0),
                optional_elements_rating=evaluation_data.get("optional_elements_rating", 0),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                response_time=response_time,
                raw_response=response
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating answer for question: {e}")
            # Return error result
            return EvaluationResult(
                question=question,
                answer=answer,
                model_name="",  # Will be set by caller
                evaluator_model=self.model,
                analysis=f"ERROR: {str(e)}",
                required_elements_rating=0,
                prohibited_elements_rating=0,
                optional_elements_rating=0,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                response_time=time.time() - start_time,
                raw_response=f"Error: {str(e)}"
            )

    def _construct_evaluation_prompt(self, question: str, answer: str, rubric: Dict[str, str]) -> str:
        """Construct the evaluation prompt for the model"""

        prompt = f"""You are an expert evaluator of answers to questions about The Church of Jesus Christ of Latter-day Saints (LDS Church). Your task is to evaluate how well an answer addresses a question based on a provided rubric.

QUESTION: {question}

ANSWER TO EVALUATE: {answer}

RUBRIC:
Required Elements: {rubric.get('required_elements', '')}
Optional Elements: {rubric.get('optional_elements', '')}
Prohibited Elements: {rubric.get('prohibited_elements', '')}

Please evaluate the answer and provide:
1. A detailed analysis of how well the answer addresses the required elements, avoids prohibited elements, and engages with optional elements
2. Three numerical ratings on a scale of 1-10 (where 10 is the highest/best rating):
   - Required Elements Rating: How well does the answer capture the required elements?
   - Prohibited Elements Rating: How well does the answer avoid the prohibited elements? (Higher rating = better avoidance)
   - Optional Elements Rating: How well does the answer engage with optional elements?

Return your response in the following JSON format:
{{
    "analysis": "Your detailed analysis of the answer...",
    "required_elements_rating": <integer 1-10>,
    "prohibited_elements_rating": <integer 1-10>,
    "optional_elements_rating": <integer 1-10>
}}"""

        return prompt

    def _get_model_response(self, prompt: str) -> str:
        """Get response from the evaluator model via OpenRouter"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.model_config.get("temperature", 0.3),
                max_tokens=self.model_config.get("max_tokens", 1500),
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error getting response from {self.model}: {e}")
            raise

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the evaluation response to extract structured data"""
        try:
            import re
            
            # Find JSON-like structures in the response
            json_pattern = r'\{[^{}]*"analysis"[^{}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Use the first match
                json_str = matches[0]
                # Clean up the JSON string
                cleaned_json = json_str.replace('\n', ' ').replace('\r', ' ')
                
                # Parse the JSON
                evaluation_data = json.loads(cleaned_json)
                
                # Validate and convert ratings to integers
                for key in ['required_elements_rating', 'prohibited_elements_rating', 'optional_elements_rating']:
                    if key in evaluation_data:
                        try:
                            evaluation_data[key] = int(evaluation_data[key])
                            # Ensure rating is within valid range
                            evaluation_data[key] = max(1, min(10, evaluation_data[key]))
                        except (ValueError, TypeError):
                            evaluation_data[key] = 0
                    else:
                        evaluation_data[key] = 0
                
                return evaluation_data
            else:
                # If no JSON found, create a basic structure
                logger.warning(f"No JSON found in evaluation response")
                return {
                    "analysis": "Unable to parse evaluation response",
                    "required_elements_rating": 0,
                    "prohibited_elements_rating": 0,
                    "optional_elements_rating": 0
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in evaluation response: {e}")
            return {
                "analysis": "JSON parsing error",
                "required_elements_rating": 0,
                "prohibited_elements_rating": 0,
                "optional_elements_rating": 0
            }
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            return {
                "analysis": "Parsing error",
                "required_elements_rating": 0,
                "prohibited_elements_rating": 0,
                "optional_elements_rating": 0
            }

class BenchmarkAnswerEvaluator:
    """Main class for evaluating answers against rubrics"""

    def __init__(self, answers_file: str, rubrics_file: str, output_dir: str = "results"):
        """Initialize the answer evaluator"""
        self.answers_file = answers_file
        self.rubrics_file = rubrics_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load answers and rubrics
        self.answers = self._load_answers()
        self.rubrics = self._load_rubrics()
        
        # Create rubric lookup by question
        self.rubric_lookup = {r["question"]: r for r in self.rubrics}
        
        logger.info(f"Loaded {len(self.answers)} answers from {answers_file}")
        logger.info(f"Loaded {len(self.rubrics)} rubrics from {rubrics_file}")

        # Initialize results storage
        self.results: List[EvaluationResult] = []
        self.completed_evaluations: set = set()

        # Load checkpoint if exists
        self._load_checkpoint()

    def _load_answers(self) -> List[Dict[str, Any]]:
        """Load answers from JSON file"""
        try:
            with open(self.answers_file, 'r', encoding='utf-8') as file:
                answers = json.load(file)
            return answers
        except Exception as e:
            logger.error(f"Failed to load answers from {self.answers_file}: {e}")
            raise

    def _load_rubrics(self) -> List[Dict[str, Any]]:
        """Load rubrics from JSON file"""
        try:
            with open(self.rubrics_file, 'r', encoding='utf-8') as file:
                rubrics = json.load(file)
            return rubrics
        except Exception as e:
            logger.error(f"Failed to load rubrics from {self.rubrics_file}: {e}")
            raise

    def _get_checkpoint_file(self, evaluator_model_name: str) -> Path:
        """Get checkpoint file path for a specific evaluator model"""
        # Replace slashes with underscores to avoid file path issues
        safe_model_name = evaluator_model_name.replace("/", "_")
        return self.output_dir / f"evaluation_checkpoint_{safe_model_name}.json"

    def _save_checkpoint(self, evaluator_model_name: str):
        """Save checkpoint for a specific evaluator model"""
        checkpoint_file = self._get_checkpoint_file(evaluator_model_name)

        checkpoint_data = {
            "evaluator_model": evaluator_model_name,
            "completed_evaluations": list(self.completed_evaluations),
            "results": [asdict(result) for result in self.results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as file:
                json.dump(checkpoint_data, file, indent=2)
            logger.info(f"Saved checkpoint for {evaluator_model_name}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        # This will be called after model initialization
        pass

    def _load_model_checkpoint(self, evaluator_model_name: str):
        """Load checkpoint for a specific evaluator model"""
        checkpoint_file = self._get_checkpoint_file(evaluator_model_name)

        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as file:
                    checkpoint_data = json.load(file)

                # Restore completed evaluations
                self.completed_evaluations = set(checkpoint_data.get("completed_evaluations", []))

                # Restore results
                self.results = []
                for result_data in checkpoint_data.get("results", []):
                    result = EvaluationResult(**result_data)
                    self.results.append(result)

                logger.info(f"Loaded checkpoint for {evaluator_model_name}: {len(self.completed_evaluations)} completed evaluations, {len(self.results)} results")

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                # Continue without checkpoint
        else:
            logger.info(f"No checkpoint found for {evaluator_model_name}, starting fresh")

    def evaluate_answers(self, evaluator_model_name: str, model_config: Dict[str, Any], checkpoint_interval: int = 10):
        """Evaluate all answers against their rubrics"""

        logger.info(f"Starting answer evaluation with {evaluator_model_name}")

        # Load checkpoint for this evaluator model
        self._load_model_checkpoint(evaluator_model_name)

        # Initialize answer evaluator
        try:
            evaluator = AnswerEvaluator(evaluator_model_name, model_config)
        except Exception as e:
            logger.error(f"Failed to initialize {evaluator_model_name}: {e}")
            return

        # Get answers that haven't been evaluated
        remaining_answers = []
        for answer in self.answers:
            question = answer["question"]
            if question not in self.completed_evaluations and question in self.rubric_lookup:
                remaining_answers.append(answer)
            elif question not in self.rubric_lookup:
                logger.warning(f"No rubric found for question: {question}")

        if not remaining_answers:
            logger.info(f"All answers already evaluated with {evaluator_model_name}")
            return

        logger.info(f"Evaluating {len(remaining_answers)} remaining answers with {evaluator_model_name}")

        # Get delay from environment or use default
        delay = float(os.getenv("DELAY_BETWEEN_EVALUATIONS", "1.0"))

        # Evaluate each answer
        for i, answer in enumerate(remaining_answers):
            question = answer["question"]
            logger.info(f"{evaluator_model_name} - Evaluating answer {i+1}/{len(remaining_answers)}: {question[:50]}...")

            try:
                # Get the rubric for this question
                rubric = self.rubric_lookup[question]

                # Evaluate the answer
                result = evaluator.evaluate_answer(question, answer["answer"], rubric)
                result.model_name = answer.get("model_name", "unknown")

                # Store result
                self.results.append(result)
                self.completed_evaluations.add(question)

                # Log result
                logger.info(f"{evaluator_model_name} - Evaluated: {question[:50]}... (R:{result.required_elements_rating}, P:{result.prohibited_elements_rating}, O:{result.optional_elements_rating})")

                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(evaluator_model_name)
                    logger.info(f"Checkpoint saved after {i+1} evaluations")

                # Add delay to avoid rate limiting
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Error evaluating answer for question: {e}")
                continue

        # Final checkpoint
        self._save_checkpoint(evaluator_model_name)
        logger.info(f"Completed answer evaluation with {evaluator_model_name}")

    def export_results(self, evaluator_model_name: str, format: str = "json"):
        """Export results to specified format"""

        if format.lower() == "json":
            self._export_json(evaluator_model_name)
        else:
            logger.error(f"Unsupported format: {format}")

    def _export_json(self, evaluator_model_name: str):
        """Export results to JSON"""

        json_file = self.output_dir / f"evaluations_{evaluator_model_name.replace('/', '_')}.json"

        try:
            # Convert results to list of dictionaries
            results_data = [asdict(result) for result in self.results]

            with open(json_file, 'w', encoding='utf-8') as file:
                json.dump(results_data, file, indent=2, ensure_ascii=False)

            logger.info(f"Exported results to {json_file}")

            # Print summary
            total_evaluations = len(self.results)
            successful_evaluations = sum(1 for r in self.results if not r.analysis.startswith("ERROR"))

            # Calculate average ratings
            if successful_evaluations > 0:
                avg_required = sum(r.required_elements_rating for r in self.results if not r.analysis.startswith("ERROR")) / successful_evaluations
                avg_prohibited = sum(r.prohibited_elements_rating for r in self.results if not r.analysis.startswith("ERROR")) / successful_evaluations
                avg_optional = sum(r.optional_elements_rating for r in self.results if not r.analysis.startswith("ERROR")) / successful_evaluations
            else:
                avg_required = avg_prohibited = avg_optional = 0

            print(f"\n📊 Evaluation Summary for {evaluator_model_name}:")
            print(f"Total Evaluations: {total_evaluations}")
            print(f"Successful Evaluations: {successful_evaluations}")
            print(f"Failed Evaluations: {total_evaluations - successful_evaluations}")
            print(f"Average Required Elements Rating: {avg_required:.2f}/10")
            print(f"Average Prohibited Elements Rating: {avg_prohibited:.2f}/10")
            print(f"Average Optional Elements Rating: {avg_optional:.2f}/10")
            print(f"Results saved to: {json_file}")

        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model using OpenRouter"""

    # Get OpenRouter API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found. Please set this environment variable.")

    # Use the model name from command line, or fall back to environment variable
    model = os.getenv("OPENROUTER_MODEL", model_name)
    temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "1500"))

    return {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

def load_environment_config() -> Dict[str, Any]:
    """Load configuration from environment variables and .env file"""

    config = {}

    # OpenRouter API Key
    config["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY")

    # Model configurations
    config["openrouter_model"] = os.getenv("OPENROUTER_MODEL")
    config["openrouter_temperature"] = float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
    config["openrouter_max_tokens"] = int(os.getenv("OPENROUTER_MAX_TOKENS", "1500"))

    # Testing configurations
    config["checkpoint_interval"] = int(os.getenv("CHECKPOINT_INTERVAL", "10"))
    config["output_directory"] = os.getenv("OUTPUT_DIRECTORY", "results")
    config["delay_between_evaluations"] = float(os.getenv("DELAY_BETWEEN_EVALUATIONS", "1.0"))

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
    print(f"   Checkpoint Interval: {config['checkpoint_interval']} evaluations")
    print(f"   Output Directory: {config['output_directory']}")
    print(f"   Delay Between Evaluations: {config['delay_between_evaluations']} seconds")
    print(f"   Log Level: {config['log_level']}")

    print("=" * 50)

def print_environment_help():
    """Print help information for environment configuration"""

    print("💡 Environment Configuration Help")
    print("=" * 50)
    print("The Faith Benchmark Answer Evaluator script can be configured using:")
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
    print("- OPENROUTER_MODEL: Model to use (default: uses --evaluator-model parameter)")
    print("- OPENROUTER_TEMPERATURE: Temperature setting (default: 0.3)")
    print("- OPENROUTER_MAX_TOKENS: Max tokens per response (default: 1500)")
    print("- CHECKPOINT_INTERVAL: Save progress every N evaluations (default: 10)")
    print("- OUTPUT_DIRECTORY: Directory for results (default: 'results')")
    print("- DELAY_BETWEEN_EVALUATIONS: Delay in seconds (default: 1.0)")
    print("- LOG_LEVEL: Logging level (default: INFO)")

    print("\n🚀 Usage Examples:")
    print("1. Check configuration status:")
    print("   python answer_evaluator.py --status")
    print("2. Evaluate answers with GPT-4 via OpenRouter:")
    print("   python answer_evaluator.py --answers answers.json --rubrics rubrics.json --evaluator-model openai/gpt-4")
    print("3. Evaluate answers with Claude via OpenRouter:")
    print("   python answer_evaluator.py --answers answers.json --rubrics rubrics.json --evaluator-model anthropic/claude-3-sonnet")
    print("4. Override environment settings:")
    print("   python answer_evaluator.py --answers answers.json --rubrics rubrics.json --evaluator-model openai/gpt-4 --checkpoint-interval 5")

    print("=" * 50)

def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Evaluate answers to open-ended faith benchmark questions using LLM evaluators")
    parser.add_argument("--answers", required=False,
                       help="Path to JSON file containing answers (generated by generate_answers.py)")
    parser.add_argument("--rubrics", required=False,
                       help="Path to JSON file containing rubrics (generated by rubric_generator.py)")
    parser.add_argument("--evaluator-model", required=False,
                       help="LLM model to use as evaluator (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results and checkpoints (overrides OUTPUT_DIRECTORY env var)")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                       help="Save checkpoint every N evaluations (overrides CHECKPOINT_INTERVAL env var)")
    parser.add_argument("--format", default="json", choices=["json"],
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

    # Check if files exist
    if not os.path.exists(args.answers):
        print(f"❌ Answers file not found: {args.answers}")
        print("Please specify the correct path to the answers JSON file.")
        sys.exit(1)

    if not os.path.exists(args.rubrics):
        print(f"❌ Rubrics file not found: {args.rubrics}")
        print("Please specify the correct path to the rubrics JSON file.")
        sys.exit(1)

    # Get model configuration
    try:
        model_config = get_model_config(args.evaluator_model)
    except ValueError as e:
        print(f"❌ {e}")
        print("\n💡 Environment Configuration Help:")
        print("   Set OpenRouter API key in environment variables or .env file:")
        print("   - OPENROUTER_API_KEY for accessing models via OpenRouter")
        print("\n   Run with --help-env for detailed configuration help")
        print("   Run with --status to see current configuration")
        sys.exit(1)

    # Initialize answer evaluator
    try:
        evaluator = BenchmarkAnswerEvaluator(args.answers, args.rubrics, output_dir)
    except Exception as e:
        print(f"❌ Failed to initialize answer evaluator: {e}")
        sys.exit(1)

    # Evaluate answers
    print(f"🚀 Starting answer evaluation with {args.evaluator_model}")
    print(f"📁 Answers file: {args.answers}")
    print(f"📁 Rubrics file: {args.rubrics}")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Checkpoint interval: {checkpoint_interval} evaluations")
    print(f"⏱️  Delay between evaluations: {env_config['delay_between_evaluations']} seconds")
    print("=" * 60)

    try:
        evaluator.evaluate_answers(args.evaluator_model, model_config, checkpoint_interval)

        # Export results
        evaluator.export_results(args.evaluator_model, args.format)

        print(f"\n✅ Answer evaluation completed with {args.evaluator_model}!")

    except KeyboardInterrupt:
        print(f"\n⚠️  Answer evaluation interrupted. Progress has been saved to checkpoint.")
        print("You can resume by running the same command again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Answer evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
