# Nutrition Agent 🥗🤖

An AI-powered meal planning assistant that generates balanced meal plans based on available ingredients, dietary constraints, and nutritional targets. This project demonstrates practical agent implementation patterns learned from the agent-engineering course.

## 🎯 Project Overview

The Nutrition Agent takes your refrigerator inventory and creates 3-day meal plans that:

- ✅ Meet your nutritional targets (calories, PFC ratio)
- ✅ Respect dietary restrictions and allergies
- ✅ Maximize use of available ingredients
- ✅ Generate shopping lists for missing items
- ✅ Provide detailed nutritional breakdowns

### Key Features

- **Multi-tool Agent Architecture**: Uses FatSecret API for nutrition data and recipe search
- **Robust Evaluation System**: Automated scoring based on nutrition accuracy and ingredient optimization
- **Interactive CLI**: User-friendly command-line interface with rich formatting
- **Extensible Design**: Easy to add new dietary constraints and evaluation metrics

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Agent
        A[Planner<br>(LLM)] -->|candidate meals| B
        B[FatSecret Nutrition Checker<br>(tool call)]
        B -->|nutrition OK| C[Formatter]
        B -->|NG→re-propose| A
    end
    User -->|inventory & targets| A
    C -->|meal plan & shopping list| User
```

The agent follows a tool-using LLM loop pattern:

1. **Planner**: LLM generates meal combinations from available ingredients
2. **Nutrition Checker**: Validates nutritional balance using FatSecret API
3. **Formatter**: Structures output as JSON with detailed breakdowns
4. **Shopping List Generator**: Identifies missing ingredients for complete meals

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- FatSecret API credentials ([Get them here](https://platform.fatsecret.com/docs/guides))
- OpenAI API key

### Installation

1. **Clone and setup**:

```bash
git clone <repository-url>
cd nutrition-agent
```

2. **Install dependencies**:

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -e .
```

3. **Configure environment**:

```bash
cp .env.example .env
# Edit .env with your API keys:
# FATSECRET_CONSUMER_KEY=your_key_here
# FATSECRET_CONSUMER_SECRET=your_secret_here
# OPENAI_API_KEY=your_openai_key_here
```

### Basic Usage

**Interactive Mode**:

```bash
python main.py interactive
```

**Sample Scenarios**:

```bash
# List available test scenarios
python main.py list-samples

# Run with sample data
python main.py sample t1 --model gpt-4.1 --days 3
```

**Evaluation**:

```bash
# Evaluate all scenarios with multiple models
python evaluate.py run --models gpt-4.1 gpt-4.1-mini

# Single scenario evaluation
python evaluate.py single t1.json --model gpt-4.1
```

## 📁 Project Structure

```
nutrition-agent/
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Base agent with tool-calling loop
│   └── nutrition_planner.py # Nutrition-specific agent
├── tools/                  # Tool implementations
│   ├── fatsecret_tool.py   # FatSecret API wrapper
│   └── nutrition_calculator.py # PFC balance calculations
├── evaluators/             # Evaluation system
│   └── nutrition_evaluator.py # Reward function implementation
├── config/                 # Configuration files
│   └── prompts.yaml       # System prompts and examples
├── data/                   # Test data and scenarios
│   ├── test_prompts/      # Test scenarios (T1, T2, T3)
│   └── ground_truth/      # Expected results for evaluation
├── tests/                  # Unit tests
├── main.py                # Main CLI interface
├── evaluate.py            # Evaluation runner
└── pyproject.toml         # Dependencies and project config
```

## 🧪 Test Scenarios

The project includes three pre-defined test scenarios:

| ID     | Description                   | Special Constraints          |
| ------ | ----------------------------- | ---------------------------- |
| **T1** | Basic single-person household | Standard omnivore diet       |
| **T2** | Vegetarian household          | Plant-based proteins only    |
| **T3** | Low-carb diet                 | Max 100g carbs/day, high fat |

Each scenario includes:

- Available ingredient inventory
- Nutritional targets (calories, PFC ratio)
- Dietary restrictions and allergens
- Ground truth for evaluation

### Sample Scenario (T1):

```json
{
  "inventory": [
    { "name": "chicken_breast", "amount_g": 400 },
    { "name": "eggs", "amount_g": 360 },
    { "name": "white_rice", "amount_g": 300 }
  ],
  "constraints": {
    "daily_calories": 2000.0,
    "pfc_ratio": [30.0, 25.0, 45.0]
  }
}
```

## 📊 Evaluation System

The evaluation system implements the reward function from `first_assignment.md`:

### Scoring Formula

```python
score = nutrition_score + shopping_list_score

# Nutrition Score (0.0 to 0.5)
if all(macro_error <= 10% for macro in [protein, fat, carbs]) and calorie_error <= 10%:
    nutrition_score = 0.5
else:
    nutrition_score = scaled_penalty

# Shopping List Score (0.0 to 0.5)
shopping_list_score = 0.5 * jaccard_similarity(predicted, ground_truth)

# Allergen Violation = Immediate 0.0
```

### Evaluation Metrics

- **Nutrition Accuracy**: PFC balance within ±10% of targets
- **Calorie Precision**: Total calories within ±10% of target
- **Shopping List Quality**: Jaccard similarity with ground truth missing ingredients
- **Allergen Safety**: Zero tolerance for allergen violations
- **Execution Time**: Performance benchmarking

### Running Evaluations

**Compare Models**:

```bash
python evaluate.py compare --models gpt-4.1 gpt-4.1-mini
```

**Detailed Analysis**:

```bash
python evaluate.py run --output detailed_results.json
```

**Validate Test Setup**:

```bash
python evaluate.py validate
```

## 🔧 Configuration

### Model Configuration

The agent supports multiple LLM providers:

```python
# OpenAI GPT models
config = AgentConfig(
    model_provider=ModelProvider.OPENAI,
    model_name="gpt-4.1",
    temperature=0.7,
    max_tokens=4000
)

```

### Custom Prompts

Edit `config/prompts.yaml` to customize:

- System prompts
- Few-shot examples
- Validation criteria
- Output format instructions

## 🚧 Implementation Roadblocks & Solutions

### Common Issues Encountered:

1. **FatSecret API Rate Limiting**

   - **Solution**: Implemented request caching and rate limiting with asyncio
   - **Code**: `tools/fatsecret_tool.py:_make_request()`

2. **Inconsistent Recipe Suggestions**

   - **Solution**: Added prompt engineering for variety and nutrition validation loops
   - **Code**: Enhanced system prompt in `config/prompts.yaml`

3. **Nutrition Calculation Accuracy**

   - **Solution**: Multiple validation passes and tolerance-based scoring
   - **Code**: `tools/nutrition_calculator.py:calculate_nutrition_error()`

4. **Model Response Parsing**
   - **Solution**: Structured output with Pydantic models and retry mechanisms
   - **Code**: `agents/base_agent.py:_get_llm_response()`

## 📈 Performance Results

### Model Comparison (Preliminary Results)

| Model               | Avg Score | Nutrition Accuracy | Shopping List | Avg Time | Cost/Run |
| ------------------- | --------- | ------------------ | ------------- | -------- | -------- |
| **GPT-4.1**         | 0.847     | 0.432/0.5          | 0.415/0.5     | 23.2s    | $0.12    |
| **GPT-4.1-mini**    | 0.723     | 0.378/0.5          | 0.345/0.5     | 8.7s     | $0.03    |

### Key Findings:

- **GPT-4.1**: Best overall performance, especially nutrition accuracy
- **GPT-4.1-mini**: Most cost-effective, acceptable performance for simpler scenarios

## 🔮 Future Enhancements

### Phase 2 Features:

- [ ] **Multi-Agent Architecture**: Separate planning, validation, and optimization agents
- [ ] **Best-of-N Selection**: Generate multiple plans and select best scoring
- [ ] **Async Parallel Processing**: Speed up API calls with concurrent requests
- [ ] **Recipe Database Integration**: Local cache for faster nutrition lookups
- [ ] **Meal Preference Learning**: Adapt to user preferences over time

### Advanced Evaluation:

- [ ] **LLM Judge Evaluation**: Taste and practicality scoring
- [ ] **Embedding Similarity**: Compare meal variety using embeddings
- [ ] **User Study Integration**: Real-world usability testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Run tests: `pytest tests/`
4. Run evaluation: `python evaluate.py validate`
5. Submit a pull request

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Agent Engineering Course**: Foundation patterns and architectural guidance
- **FatSecret Platform**: Nutrition database API
- **OpenAI & Anthropic**: LLM capabilities that make this agent possible
