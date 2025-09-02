#!/usr/bin/env python3
"""
Example usage of the Faith Benchmark Question Generator
This script demonstrates how to use the generator programmatically
"""

import os
from question_generator import FaithBenchmarkQuestionGenerator

def example_basic_usage():
    """Basic example of generating questions for a few topics"""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("🚀 Faith Benchmark Question Generator - Example Usage")
    print("=" * 60)
    
    # Initialize generator
    generator = FaithBenchmarkQuestionGenerator(api_key)
    
    # Load topics
    topics = generator.load_topics()
    if not topics:
        print("❌ No topics found")
        return
    
    print(f"📚 Loaded {len(topics)} topics")
    
    # Select a few interesting topics for demonstration
    sample_topics = ["Baptism", "Book of Mormon", "Joseph Smith"]
    available_topics = [t for t in sample_topics if t in topics]
    
    if not available_topics:
        print("❌ None of the sample topics found in the topics list")
        return
    
    print(f"\n🎯 Generating questions for: {', '.join(available_topics)}")
    
    # Generate questions
    questions = generator.generate_all_questions(available_topics, questions_per_topic=2)
    
    if questions:
        # Save results
        filename = "example_questions.json"
        generator.save_questions(questions, filename)
        
        print(f"\n✅ Generated {len(questions)} questions!")
        print(f"📁 Saved to: {filename}")
        
        # Display results
        print("\n📝 Generated Questions:")
        for i, q in enumerate(questions):
            print(f"\n{i+1}. Topic: {q.topic}")
            print(f"   Question: {q.question_text}")
            print(f"   Correct Answer: {q.correct_answer}")
            print(f"   Tags: {', '.join(q.tags)}")
            print(f"   Difficulty: {q.difficulty}")
            print(f"   Source Tier: {q.source_tier}")
            print(f"   Explanation: {q.explanation}")
    else:
        print("❌ No questions generated")

def example_custom_prompt():
    """Example of customizing the generation prompt"""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("\n🔧 Custom Prompt Example")
    print("=" * 40)
    
    # Create a custom generator with modified prompt
    generator = FaithBenchmarkQuestionGenerator(api_key)
    
    # Override the system prompt for a specific topic
    original_method = generator.generate_questions_for_topic
    
    def custom_generate_questions_for_topic(topic, num_questions=3):
        """Custom method with modified prompt"""
        
        custom_system_prompt = """You are an expert in religious studies focused on The Church of Jesus Christ of Latter-day Saints.

Generate questions that are specifically designed for BEGINNERS who are learning about LDS beliefs. Focus on:
- Basic, fundamental concepts
- Simple, clear explanations
- Common misconceptions
- Everyday practices

Avoid complex theological debates or advanced historical details."""

        user_prompt = f"""Generate {num_questions} BEGINNER-LEVEL questions about: "{topic}"

Focus on basic understanding and common knowledge. Provide JSON format:
{{
    "question_text": "Simple, clear question",
    "answers": ["Answer A", "Answer B", "Answer C", "Answer D"],
    "correct_answer": "Answer A",
    "explanation": "Simple explanation",
    "tags": ["beginner", "basic", "fundamental"],
    "source_tier": "A",
    "difficulty": "basic"
}}"""

        try:
            response = generator.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": custom_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            return generator._parse_openai_response(content, topic)
            
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    # Use custom method
    generator.generate_questions_for_topic = custom_generate_questions_for_topic
    
    # Generate beginner questions
    topics = ["Baptism", "Prayer"]
    questions = generator.generate_all_questions(topics, questions_per_topic=1)
    
    if questions:
        filename = "beginner_questions.json"
        generator.save_questions(questions, filename)
        print(f"✅ Generated {len(questions)} beginner questions")
        print(f"📁 Saved to: {filename}")
        
        # Show sample
        for q in questions:
            print(f"\n📚 {q.topic}: {q.question_text}")
            print(f"   Difficulty: {q.difficulty}")
            print(f"   Tags: {', '.join(q.tags)}")

if __name__ == "__main__":
    print("Faith Benchmark Question Generator - Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_custom_prompt()
    
    print("\n✨ Examples completed!")
    print("\nTo run the full interactive system, use: python main.py")
