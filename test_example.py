#!/usr/bin/env python3
"""
Example usage of the Faith Benchmark Test Accuracy Script
This script demonstrates how to test models programmatically
"""

import os
import sys
from test_accuracy import BenchmarkTester, get_model_config

def example_basic_testing():
    """Basic example of testing a model on benchmark questions"""
    
    # Check for questions file
    questions_file = "faith_benchmark_questions.json"
    if not os.path.exists(questions_file):
        print(f"❌ Questions file not found: {questions_file}")
        print("Please run the question generator first to create questions.")
        return
    
    # Check for API keys
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GOOGLE_API_KEY"),
        "claude": os.getenv("ANTHROPIC_API_KEY")
    }
    
    available_models = [model for model, key in api_keys.items() if key]
    
    if not available_models:
        print("❌ No API keys found!")
        print("Please set at least one of:")
        print("- OPENAI_API_KEY for OpenAI models")
        print("- GOOGLE_API_KEY for Gemini models")
        print("- ANTHROPIC_API_KEY for Claude models")
        return
    
    print("🚀 Faith Benchmark Test Accuracy - Example Usage")
    print("=" * 60)
    print(f"📁 Questions file: {questions_file}")
    print(f"🔑 Available models: {', '.join(available_models)}")
    
    # Initialize benchmark tester
    try:
        tester = BenchmarkTester(questions_file, "example_results")
        print("✅ Benchmark tester initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize benchmark tester: {e}")
        return
    
    # Test with first available model
    model_name = available_models[0]
    print(f"\n🧪 Testing with {model_name} model...")
    
    try:
        # Get model configuration
        model_config = get_model_config(model_name)
        
        # Run tests (limit to first 10 questions for example)
        print("Note: Limiting to first 10 questions for this example")
        original_questions = tester.questions
        tester.questions = original_questions[:10]
        
        # Run tests
        tester.run_tests(model_name, model_config, checkpoint_interval=5)
        
        # Export results
        tester.export_results(model_name, "csv")
        
        print(f"\n✅ Testing completed for {model_name}!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return

def example_resume_from_checkpoint():
    """Example of resuming from a checkpoint"""
    
    questions_file = "faith_benchmark_questions.json"
    if not os.path.exists(questions_file):
        print(f"❌ Questions file not found: {questions_file}")
        return
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return
    
    print("\n🔄 Resume from Checkpoint Example")
    print("=" * 40)
    
    try:
        # Initialize tester
        tester = BenchmarkTester(questions_file, "example_results")
        
        # Check if checkpoint exists
        checkpoint_file = tester._get_checkpoint_file("openai")
        if checkpoint_file.exists():
            print(f"📁 Found checkpoint: {checkpoint_file}")
            
            # Load checkpoint
            tester._load_model_checkpoint("openai")
            
            print(f"📊 Checkpoint loaded:")
            print(f"   Completed questions: {len(tester.completed_questions)}")
            print(f"   Results: {len(tester.results)}")
            
            # Continue testing
            model_config = get_model_config("openai")
            tester.run_tests("openai", model_config, checkpoint_interval=5)
            
        else:
            print("📁 No checkpoint found, starting fresh")
            model_config = get_model_config("openai")
            tester.run_tests("openai", model_config, checkpoint_interval=5)
        
        # Export results
        tester.export_results("openai", "csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_custom_testing():
    """Example of custom testing configuration"""
    
    questions_file = "faith_benchmark_questions.json"
    if not os.path.exists(questions_file):
        print(f"❌ Questions file not found: {questions_file}")
        return
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return
    
    print("\n⚙️  Custom Testing Configuration Example")
    print("=" * 40)
    
    try:
        # Initialize tester with custom output directory
        tester = BenchmarkTester(questions_file, "custom_results")
        
        # Custom model configuration
        custom_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4"  # You could change this to gpt-3.5-turbo
        }
        
        # Test with custom checkpoint interval
        checkpoint_interval = 3  # Save every 3 questions
        print(f"💾 Checkpoint interval: {checkpoint_interval}")
        
        # Run tests
        tester.run_tests("openai", custom_config, checkpoint_interval)
        
        # Export results
        tester.export_results("openai", "csv")
        
        print("✅ Custom testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Faith Benchmark Test Accuracy - Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_testing()
    example_resume_from_checkpoint()
    example_custom_testing()
    
    print("\n✨ Examples completed!")
    print("\nTo run the full command-line testing, use:")
    print("python test_accuracy.py --model openai --questions faith_benchmark_questions.json")
