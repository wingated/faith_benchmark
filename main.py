#!/usr/bin/env python3
"""
Faith Benchmark Question Generator
Main script for generating questions using OpenAI API
"""

import os
import sys
from question_generator import FaithBenchmarkQuestionGenerator

def setup_environment():
    """Set up environment variables and check configuration"""
    # Try to load from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set the OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("OPENAI_API_KEY=your-api-key-here")
        return None
    
    return api_key

def main():
    """Main function with user interface"""
    print("🏛️  Faith Benchmark Question Generator")
    print("=" * 50)
    
    # Check environment
    api_key = setup_environment()
    if not api_key:
        sys.exit(1)
    
    # Initialize generator
    try:
        generator = FaithBenchmarkQuestionGenerator(api_key)
        print("✅ Generator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        sys.exit(1)
    
    # Load topics
    topics = generator.load_topics()
    if not topics:
        print("❌ No topics found. Please check the list_of_topics.txt file.")
        sys.exit(1)
    
    print(f"📚 Found {len(topics)} topics")
    
    # Get user preferences
    print("\nOptions:")
    print("1. Generate questions for all topics (this will take a while)")
    print("2. Generate questions for first 5 topics (for testing)")
    print("3. Generate questions for specific number of topics")
    print("4. Generate questions for specific topics")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # All topics
        print(f"\n🚀 Generating questions for all {len(topics)} topics...")
        print("This will take a while and may incur significant API costs.")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            questions = generator.generate_all_questions(topics, questions_per_topic=3)
            filename = "faith_benchmark_questions_complete.json"
        else:
            print("Operation cancelled.")
            return
    
    elif choice == "2":
        # Test with first 5 topics
        print("\n🧪 Generating questions for first 5 topics (testing mode)...")
        test_topics = topics[:5]
        questions = generator.generate_all_questions(test_topics, questions_per_topic=3)
        filename = "faith_benchmark_questions_test.json"
    
    elif choice == "3":
        # Specific number of topics
        try:
            num_topics = int(input("Enter number of topics to process: "))
            if num_topics > len(topics):
                print(f"⚠️  Only {len(topics)} topics available. Using all topics.")
                num_topics = len(topics)
            
            print(f"\n🚀 Generating questions for first {num_topics} topics...")
            selected_topics = topics[:num_topics]
            questions = generator.generate_all_questions(selected_topics, questions_per_topic=3)
            filename = f"faith_benchmark_questions_{num_topics}topics.json"
        except ValueError:
            print("❌ Invalid number. Please enter a valid integer.")
            return
    
    elif choice == "4":
        # Specific topics
        print("\nAvailable topics:")
        for i, topic in enumerate(topics[:20]):  # Show first 20
            print(f"{i+1:2d}. {topic}")
        if len(topics) > 20:
            print(f"... and {len(topics) - 20} more")
        
        topic_input = input("\nEnter topic numbers separated by commas (e.g., 1,3,5): ").strip()
        try:
            topic_indices = [int(x.strip()) - 1 for x in topic_input.split(',')]
            selected_topics = [topics[i] for i in topic_indices if 0 <= i < len(topics)]
            
            if not selected_topics:
                print("❌ No valid topics selected.")
                return
            
            print(f"\n🚀 Generating questions for {len(selected_topics)} selected topics...")
            questions = generator.generate_all_questions(selected_topics, questions_per_topic=3)
            filename = "faith_benchmark_questions_selected.json"
        except ValueError:
            print("❌ Invalid input. Please enter valid topic numbers.")
            return
    
    else:
        print("❌ Invalid choice.")
        return
    
    # Save results
    if questions:
        generator.save_questions(questions, filename)
        print(f"\n✅ Successfully generated {len(questions)} questions!")
        print(f"📁 Saved to: {filename}")
        
        # Show sample
        print("\n📝 Sample questions:")
        for i, q in enumerate(questions[:3]):
            print(f"\n{i+1}. {q.question_text}")
            print(f"   Correct: {q.correct_answer}")
            print(f"   Tags: {', '.join(q.tags)}")
            print(f"   Difficulty: {q.difficulty}")
            print(f"   Source Tier: {q.source_tier}")
    else:
        print("❌ No questions were generated. Please check the API key and try again.")

if __name__ == "__main__":
    main()
