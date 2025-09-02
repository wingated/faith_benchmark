# Faith Benchmark Question Generator

This system generates factual, verifiable questions for testing AI understanding and respect of religious beliefs, specifically focused on The Church of Jesus Christ of Latter-day Saints (LDS Church).

## Overview

The Faith Benchmark Question Generator creates structured questions that test:
- **Doctrinal understanding** - Core beliefs and teachings
- **Historical knowledge** - Church history and events
- **Practical knowledge** - Practices, policies, and procedures
- **Scholarly understanding** - Academic and theological concepts

Each question includes:
- Multiple choice answers with one correct answer
- Explanations for the correct answer
- Appropriate tags for categorization
- Source tier classification (A=canonical, B=authorized, C=scholarship)
- Difficulty level (basic, intermediate, advanced)

## Features

- **Automated Generation**: Uses OpenAI API to generate contextually appropriate questions
- **Structured Output**: Each question is formatted as a JSON object with consistent fields
- **Progress Tracking**: Saves progress during generation to avoid data loss
- **Flexible Processing**: Can generate questions for all topics or specific subsets
- **Quality Control**: Questions are designed to be factual and verifiable from authoritative sources

## Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
   ```bash
   # For OpenAI
   export OPENAI_API_KEY='your-openai-api-key'
   
   # For Google Gemini
   export GOOGLE_API_KEY='your-google-api-key'
   
   # For Anthropic Claude
   export ANTHROPIC_API_KEY='your-anthropic-api-key'
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-openai-api-key
   GOOGLE_API_KEY=your-google-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

## Usage

### Quick Start

Run the main script for an interactive experience:
```bash
python main.py
```

### Programmatic Usage

```python
from question_generator import FaithBenchmarkQuestionGenerator

# Initialize generator
generator = FaithBenchmarkQuestionGenerator(api_key="your-key")

# Load topics
topics = generator.load_topics()

# Generate questions for specific topics
questions = generator.generate_all_questions(topics[:5], questions_per_topic=3)

# Save results
generator.save_questions(questions, "my_questions.json")
```

### Command Line Options

The main script provides several options:

1. **Generate for all topics** - Creates questions for all 200+ topics (time-consuming)
2. **Test mode** - Generates questions for first 5 topics (recommended for testing)
3. **Custom number** - Specify how many topics to process
4. **Specific topics** - Choose specific topics by number

## Testing Language Models

Once you have generated questions, you can test different language models on the benchmark using the `test_accuracy.py` script.

### Supported Models

- **OpenAI** (GPT-4, GPT-3.5-turbo)
- **Google Gemini** (Gemini Pro)
- **Anthropic Claude** (Claude 3 Sonnet)

### Running Tests

```bash
# Test OpenAI GPT-4
python test_accuracy.py --model openai --questions faith_benchmark_questions.json

# Test Google Gemini
python test_accuracy.py --model gemini --questions faith_benchmark_questions.json

# Test Anthropic Claude
python test_accuracy.py --model claude --questions faith_benchmark_questions.json
```

### Test Options

- `--model`: Choose the language model to test
- `--questions`: Path to the questions JSON file
- `--output-dir`: Directory for results and checkpoints (default: "results")
- `--checkpoint-interval`: Save checkpoint every N questions (default: 10)
- `--format`: Output format (currently only CSV supported)

### Checkpointing

The test script automatically saves progress every N questions (configurable). If interrupted, you can restart and it will continue from where it left off.

### Output

Results are exported to CSV files in the output directory:
- `results_openai.csv` - OpenAI test results
- `results_gemini.csv` - Gemini test results  
- `results_claude.csv` - Claude test results

Each CSV includes:
- Question details (ID, topic, text, tags)
- Correct answer and model's answer
- Accuracy (correct/incorrect)
- Response time and timestamp
- Raw model response

## Output Format

Questions are saved as JSON with this structure:

```json
{
  "id": "topic_name_1",
  "topic": "Topic Name",
  "question_text": "What is the question?",
  "question_type": "multiple_choice",
  "tags": ["doctrine", "scripture", "basic"],
  "answers": ["Answer A", "Answer B", "Answer C", "Answer D"],
  "correct_answer": "Answer A",
  "explanation": "Explanation of why this is correct",
  "source_tier": "A",
  "difficulty": "basic"
}
```

## Source Tiers

- **Tier A**: Canonical & official (scriptures, official handbooks, catechisms)
- **Tier B**: Authorized explanatory (educational materials, official histories)
- **Tier C**: Reputable neutral scholarship (peer-reviewed academic sources)

## Question Categories

Questions are tagged with relevant categories:
- `doctrine` - Core beliefs and teachings
- `history` - Historical events and figures
- `practice` - Current practices and policies
- `scripture` - Biblical and LDS scripture references
- `basic`/`intermediate`/`advanced` - Difficulty levels

## Cost Considerations

- **API Costs**: Generating questions and testing models will incur API costs
- **Rate Limiting**: Built-in delays prevent API rate limiting
- **Progress Saving**: Progress is saved regularly to avoid data loss

## Testing

Start with the test mode to verify:
1. API keys are working
2. Question quality meets expectations
3. Output format is correct
4. Model testing works as expected

## Customization

### Modifying Question Types

Edit the `system_prompt` in `generate_questions_for_topic()` to change:
- Question style and tone
- Specific requirements
- Avoided topics or approaches

### Adding New Models

To add support for new language models in `test_accuracy.py`:
1. Add the model to the `get_model_config()` function
2. Implement the model-specific logic in `_initialize_model()` and `_get_model_response()`
3. Update the requirements.txt with necessary packages

### Adding New Fields

Modify the `Question` dataclass to add new fields:
```python
@dataclass
class Question:
    # ... existing fields ...
    new_field: str = ""
```

### Changing Output Format

Modify the `save_questions()` method to change output format or add additional processing.

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure appropriate environment variables are set
2. **No Topics Loaded**: Check `list_of_topics.txt` exists and is readable
3. **API Rate Limits**: The system includes delays, but you may need to increase them
4. **JSON Parsing Errors**: Some API responses may not parse cleanly - check logs
5. **Model Import Errors**: Ensure all required packages are installed

### Debug Mode

Enable detailed logging by modifying the logging level in the scripts:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To improve the system:
1. Test with different topics and models
2. Refine the system prompt for better question quality
3. Add validation for generated questions
4. Implement question filtering or scoring
5. Add support for additional language models

## License

This project is designed for educational and research purposes related to AI faith benchmarks.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Test with a small subset of topics first
4. Verify API keys and permissions
