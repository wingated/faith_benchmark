#!/usr/bin/env python3
"""
Batch Testing Script for Faith Benchmark
Tests multiple language models in sequence on the same benchmark questions
"""

import os
import sys
import time
from pathlib import Path
from test_accuracy import BenchmarkTester, get_model_config

def get_available_models():
    """Get list of available models based on API keys"""
    
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GOOGLE_API_KEY"),
        "claude": os.getenv("ANTHROPIC_API_KEY")
    }
    
    available = [model for model, key in api_keys.items() if key]
    
    if not available:
        print("❌ No API keys found!")
        print("Please set at least one of:")
        print("- OPENAI_API_KEY for OpenAI models")
        print("- GOOGLE_API_KEY for Gemini models")
        print("- ANTHROPIC_API_KEY for Claude models")
        return []
    
    return available

def run_batch_tests(questions_file: str, models: list, output_dir: str = "batch_results", 
                    checkpoint_interval: int = 10, delay_between_models: int = 5):
    """Run tests for multiple models in sequence"""
    
    print("🚀 Faith Benchmark Batch Testing")
    print("=" * 50)
    print(f"📁 Questions file: {questions_file}")
    print(f"🤖 Models to test: {', '.join(models)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Checkpoint interval: {checkpoint_interval}")
    print(f"⏱️  Delay between models: {delay_between_models} seconds")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Track overall progress
    total_models = len(models)
    completed_models = []
    failed_models = []
    
    for i, model_name in enumerate(models):
        print(f"\n🤖 Testing model {i+1}/{total_models}: {model_name}")
        print("-" * 40)
        
        try:
            # Get model configuration
            model_config = get_model_config(model_name)
            
            # Initialize tester for this model
            tester = BenchmarkTester(questions_file, output_dir)
            
            # Run tests
            start_time = time.time()
            tester.run_tests(model_name, model_config, checkpoint_interval)
            
            # Export results
            tester.export_results(model_name, "csv")
            
            # Calculate time taken
            time_taken = time.time() - start_time
            
            print(f"✅ {model_name} completed in {time_taken:.1f} seconds")
            completed_models.append(model_name)
            
            # Add delay between models (except for the last one)
            if i < total_models - 1:
                print(f"⏳ Waiting {delay_between_models} seconds before next model...")
                time.sleep(delay_between_models)
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            failed_models.append(model_name)
            continue
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Batch Testing Summary")
    print("=" * 50)
    print(f"✅ Completed: {len(completed_models)} models")
    print(f"❌ Failed: {len(failed_models)} models")
    
    if completed_models:
        print(f"\n✅ Successfully tested: {', '.join(completed_models)}")
        print(f"📁 Results saved to: {output_dir}")
    
    if failed_models:
        print(f"\n❌ Failed models: {', '.join(failed_models)}")
        print("Check the logs above for error details.")
    
    return completed_models, failed_models

def main():
    """Main function"""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python batch_test.py <questions_file> [models...]")
        print("\nExamples:")
        print("  python batch_test.py faith_benchmark_questions.json")
        print("  python batch_test.py questions.json openai gemini")
        print("  python batch_test.py questions.json openai claude")
        return
    
    questions_file = sys.argv[1]
    
    # Check if questions file exists
    if not os.path.exists(questions_file):
        print(f"❌ Questions file not found: {questions_file}")
        return
    
    # Get models to test
    if len(sys.argv) > 2:
        # Use command line specified models
        models = sys.argv[2:]
        
        # Validate model names
        valid_models = ["openai", "gemini", "claude"]
        invalid_models = [m for m in models if m.lower() not in valid_models]
        
        if invalid_models:
            print(f"❌ Invalid model names: {', '.join(invalid_models)}")
            print(f"Valid models: {', '.join(valid_models)}")
            return
        
        models = [m.lower() for m in models]
        
    else:
        # Use all available models
        models = get_available_models()
        if not models:
            return
    
    # Check if all requested models have API keys
    missing_keys = []
    for model in models:
        if model == "openai" and not os.getenv("OPENAI_API_KEY"):
            missing_keys.append("OPENAI_API_KEY")
        elif model == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            missing_keys.append("GOOGLE_API_KEY")
        elif model == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
            missing_keys.append("ANTHROPIC_API_KEY")
    
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("Please set the required environment variables.")
        return
    
    # Configuration
    output_dir = "batch_results"
    checkpoint_interval = 10
    delay_between_models = 5
    
    # Run batch tests
    try:
        completed, failed = run_batch_tests(
            questions_file, models, output_dir, 
            checkpoint_interval, delay_between_models
        )
        
        if completed:
            print(f"\n🎉 Batch testing completed successfully!")
            print(f"📊 Tested {len(completed)} models on {questions_file}")
        else:
            print(f"\n💥 All models failed. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Batch testing interrupted by user.")
        print("Progress has been saved to checkpoints.")
        print("You can resume individual models using test_accuracy.py")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Batch testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
