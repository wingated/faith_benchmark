#!/usr/bin/env python3
"""
Script to visualize faith benchmark questions from JSON file.
Generates a static HTML file with a three-pane interface:
- Left: List of topics
- Middle: Questions for selected topic
- Right: Question details
"""

import json
import os
from collections import defaultdict

def load_questions(json_file_path):
    """Load questions from JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file_path}': {e}")
        return None

def organize_questions_by_topic(questions):
    """Organize questions by topic."""
    topics = defaultdict(list)
    for question in questions:
        topics[question['topic']].append(question)
    return dict(topics)

def generate_html(questions_by_topic, output_file_path):
    """Generate the HTML file with the three-pane interface."""
    
    # Get sorted list of topics
    topics = sorted(questions_by_topic.keys())
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Faith Benchmark Questions Browser</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
        }}
        
        .pane {{
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #ddd;
        }}
        
        .topics-pane {{
            width: 250px;
            background-color: #fff;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }}
        
        .questions-pane {{
            width: 350px;
            background-color: #fafafa;
        }}
        
        .details-pane {{
            flex: 1;
            background-color: #fff;
        }}
        
        .topic-item {{
            padding: 12px 16px;
            margin: 4px 0;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
        }}
        
        .topic-item:hover {{
            background-color: #e9ecef;
            border-color: #dee2e6;
            transform: translateY(-1px);
        }}
        
        .topic-item.active {{
            background-color: #007bff;
            color: white;
            border-color: #0056b3;
        }}
        
        .question-item {{
            padding: 16px;
            margin: 8px 0;
            background-color: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .question-item:hover {{
            background-color: #f8f9fa;
            border-color: #007bff;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        
        .question-item.active {{
            background-color: #e3f2fd;
            border-color: #2196f3;
        }}
        
        .question-text {{
            font-weight: 500;
            color: #333;
            margin-bottom: 8px;
            line-height: 1.4;
        }}
        
        .question-meta {{
            font-size: 0.85em;
            color: #666;
        }}
        
        .details-content {{
            padding: 20px;
        }}
        
        .question-detail {{
            margin-bottom: 24px;
        }}
        
        .question-detail h3 {{
            color: #333;
            margin-bottom: 16px;
            font-size: 1.4em;
            line-height: 1.3;
        }}
        
        .answers-section {{
            margin: 20px 0;
        }}
        
        .answer-option {{
            padding: 12px 16px;
            margin: 8px 0;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            background-color: #f8f9fa;
            transition: all 0.2s ease;
        }}
        
        .answer-option.correct {{
            border-color: #28a745;
            background-color: #d4edda;
            color: #155724;
        }}
        
        .answer-option.incorrect {{
            border-color: #dc3545;
            background-color: #f8d7da;
            color: #721c24;
        }}
        
        .tags-section {{
            margin: 20px 0;
        }}
        
        .tag {{
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            background-color: #e9ecef;
            color: #495057;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        
        .explanation {{
            background-color: #f8f9fa;
            padding: 16px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
            margin: 20px 0;
        }}
        
        .meta-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin: 20px 0;
        }}
        
        .meta-item {{
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }}
        
        .meta-label {{
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .meta-value {{
            font-weight: 600;
            color: #333;
        }}
        
        .header {{
            background-color: #007bff;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .header h1 {{
            font-size: 1.8em;
            margin-bottom: 8px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .no-selection {{
            text-align: center;
            color: #666;
            margin-top: 100px;
            font-size: 1.2em;
        }}
        
        .stats {{
            background-color: #f8f9fa;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-top: 12px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 1.5em;
            font-weight: 600;
            color: #007bff;
        }}
        
        .stat-label {{
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Faith Benchmark Questions Browser</h1>
        <p>Browse questions by topic and explore detailed information</p>
    </div>
    
    <!-- Hidden data element containing all questions -->
    <script id="questions-data" type="application/json">
        {json.dumps(questions_by_topic)}
    </script>
    
    <div class="container">
        <div class="pane topics-pane">
            <div class="stats">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number">{len(topics)}</div>
                        <div class="stat-label">Topics</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{sum(len(questions) for questions in questions_by_topic.values())}</div>
                        <div class="stat-label">Questions</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">{len(set(tag for questions in questions_by_topic.values() for q in questions for tag in q.get('tags', [])))}</div>
                        <div class="stat-label">Tags</div>
                    </div>
                </div>
            </div>
            
            <h3 style="margin-bottom: 16px; color: #333;">Topics</h3>
            <div id="topics-list">
"""

    # Add topics
    for topic in topics:
        question_count = len(questions_by_topic[topic])
        html_content += f"""
                <div class="topic-item" onclick="selectTopic('{topic}', this)">
                    <div style="font-weight: 600;">{topic}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 4px;">{question_count} question{'s' if question_count != 1 else ''}</div>
                </div>"""

    html_content += """
            </div>
        </div>
        
        <div class="pane questions-pane">
            <div id="questions-list">
                <div class="no-selection">
                    Select a topic to view questions
                </div>
            </div>
        </div>
        
        <div class="pane details-pane">
            <div id="question-details">
                <div class="no-selection">
                    Select a question to view details
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTopic = null;
        let currentQuestion = null;
        
        function selectTopic(topic, element) {
            // Update active state
            document.querySelectorAll('.topic-item').forEach(item => item.classList.remove('active'));
            element.classList.add('active');
            
            currentTopic = topic;
            loadQuestions(topic);
        }
        
        function loadQuestions(topic) {{
            const questions = JSON.parse(document.getElementById('questions-data').textContent);
            const topicQuestions = questions[topic] || [];
            
            const questionsList = document.getElementById('questions-list');
            questionsList.innerHTML = '';
            
            if (topicQuestions.length === 0) {{
                questionsList.innerHTML = '<div class="no-selection">No questions found for this topic</div>';
                return;
            }}
            
            topicQuestions.forEach(question => {{
                const questionDiv = document.createElement('div');
                questionDiv.className = 'question-item';
                questionDiv.onclick = function() {{
                    selectQuestion(question, this);
                }};
                
                const tags = question.tags ? question.tags.slice(0, 3).join(', ') : '';
                const difficulty = question.difficulty || 'N/A';
                
                questionDiv.innerHTML = `
                    <div class="question-text">${question.question_text}</div>
                    <div class="question-meta">
                        <div>Difficulty: ${difficulty}</div>
                        <div>Tags: ${tags}</div>
                    </div>
                `;
                
                questionsList.appendChild(questionDiv);
            }});
            
            // Clear question details
            document.getElementById('question-details').innerHTML = '<div class="no-selection">Select a question to view details</div>';
        }}
        
        function selectQuestion(question, element) {{
            // Update active state
            document.querySelectorAll('.question-item').forEach(item => item.classList.remove('active'));
            element.classList.add('active');
            
            currentQuestion = question;
            showQuestionDetails(question);
        }}
        
        function showQuestionDetails(question) {{
            const detailsDiv = document.getElementById('question-details');
            
            const answers = question.answers || [];
            const correctAnswer = question.correct_answer;
            const tags = question.tags || [];
            const explanation = question.explanation || 'No explanation provided.';
            const sourceTier = question.source_tier || 'N/A';
            const difficulty = question.difficulty || 'N/A';
            const questionType = question.question_type || 'N/A';
            
            let answersHtml = '';
            answers.forEach(answer => {{
                const isCorrect = answer === correctAnswer;
                const answerClass = isCorrect ? 'correct' : 'incorrect';
                answersHtml += `<div class="answer-option ${answerClass}">${answer}</div>`;
            }});
            
            let tagsHtml = '';
            tags.forEach(tag => {{
                tagsHtml += `<span class="tag">${tag}</span>`;
            }});
            
            detailsDiv.innerHTML = `
                <div class="details-content">
                    <div class="question-detail">
                        <h3>${question.question_text}</h3>
                        
                        <div class="meta-info">
                            <div class="meta-item">
                                <div class="meta-label">Difficulty</div>
                                <div class="meta-value">${difficulty}</div>
                            </div>
                            <div class="meta-item">
                                <div class="meta-label">Source Tier</div>
                                <div class="meta-value">${sourceTier}</div>
                            </div>
                            <div class="meta-item">
                                <div class="meta-label">Type</div>
                                <div class="meta-value">${questionType}</div>
                            </div>
                            <div class="meta-item">
                                <div class="meta-label">ID</div>
                                <div class="meta-value">${question.id}</div>
                            </div>
                        </div>
                        
                        <div class="answers-section">
                            <h4 style="margin-bottom: 12px; color: #333;">Answer Options:</h4>
                            ${answersHtml}
                        </div>
                        
                        <div class="explanation">
                            <h4 style="margin-bottom: 8px; color: #333;">Explanation:</h4>
                            <p>${explanation}</p>
                        </div>
                        
                        <div class="tags-section">
                            <h4 style="margin-bottom: 8px; color: #333;">Tags:</h4>
                            ${tagsHtml}
                        </div>
                    </div>
                </div>
            `;
        }}
        
        // Initialize with first topic if available
        window.onload = function() {{
            const firstTopic = document.querySelector('.topic-item');
            if (firstTopic) {{
                firstTopic.click();
            }}
        }};
    </script>
</body>
</html>"""

    # Write the HTML file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML file generated successfully: {output_file_path}")
        return True
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        return False

def main():
    """Main function to run the script."""
    json_file = "faith_benchmark_questions.json"
    output_file = "faith_questions_browser.html"
    
    print("Loading questions from JSON file...")
    questions = load_questions(json_file)
    
    if questions is None:
        return
    
    print(f"Loaded {len(questions)} questions")
    
    print("Organizing questions by topic...")
    questions_by_topic = organize_questions_by_topic(questions)
    
    print(f"Found {len(questions_by_topic)} topics")
    
    print("Generating HTML file...")
    success = generate_html(questions_by_topic, output_file)
    
    if success:
        print(f"\nSuccess! Open '{output_file}' in your web browser to view the questions.")
        print("The interface includes:")
        print("- Left pane: List of all topics")
        print("- Middle pane: Questions for the selected topic")
        print("- Right pane: Detailed view of the selected question")
    else:
        print("Failed to generate HTML file.")

if __name__ == "__main__":
    main()
