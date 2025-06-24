# Nutrition Agent 🥗🤖

An AI-powered meal planning assistant that generates balanced meal plans based on available ingredients, dietary constraints, and nutritional targets.

## Project Overview

The Nutrition Agent takes your refrigerator inventory and creates 3-day meal plans that:

- ✅ Meet your nutritional targets (calories, PFC ratio)
- ✅ Respect dietary restrictions and allergies
- ✅ Maximize use of available ingredients
- ✅ Generate list of missing items
- ✅ Provide detailed nutritional breakdowns

### Key Features

- **Multi-tool Agent Architecture**: Uses FatSecret API for nutrition data and recipe search
- **Robust Evaluation System**: Automated scoring based on nutrition accuracy and ingredient optimization
- **Extensible Design**: Easy to add new dietary constraints and evaluation metrics

## Architecture

The agent follows a tool-using LLM loop pattern:

1. **Planner**: LLM generates meal combinations from available ingredients
2. **Nutrition Checker**: Validates nutritional balance using FatSecret API
3. **Formatter**: Structures output as JSON with detailed breakdowns
4. **Shopping List Generator**: Identifies missing ingredients for complete meals

## Quick Start

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
uv sync
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

## Project Structure

```
nutrition-agent/
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Base agent with tool-calling loop
│   └── nutrition_planner.py # Nutrition-specific agent
├── tools/                  # Tool implementations
│   ├── fatsecret_tool.py   # FatSecret API wrapper
│   └── nutrition_calculator.py # PFC balance calculations
├── evaluators/             # Evaluation system & Reward function implementation
│   ├── evaluator_manager.py # Evaluation orchestrator
│   └── reward_functions/
│   　   ├── base.py   # Base reward function
│   　   ├── nutrition.py  # Nutrition reward
│   　   ├── constraint.py # Dietary constraints reward
│   　   ├── inventory.py  # Inventory reward
│   　   └── quality.py    # Various recipe quality reward
├── config/                 # Configuration files
│   └── prompts.yaml       # System prompts and examples
├── data/                   # Test data and scenarios
│   └── test_prompts/      # Test scenarios (T1, T2, T3, T4)
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks for demos
│   ├── comprehensive_evaluation_demo.ipynb
│   └── nutrition_agent_demo.ipynb
├── main.py                # Main CLI interface
├── evaluate.py            # Evaluation runner
├── pyproject.toml         # Dependencies and project config
├── uv.lock                # Lock file for uv package manager
└── Makefile               # Build automation
```

## Test Scenarios

The project includes four pre-defined test scenarios:

| ID     | Description                   | Special Constraints          |
| ------ | ----------------------------- | ---------------------------- |
| **T1** | Basic single-person household | Standard omnivore diet       |
| **T2** | Vegetarian household          | Plant-based proteins only    |
| **T3** | Low-carb diet                 | Max 100g carbs/day, high fat |
| **T4** | Pregnancy diet                | Special dietary requirements |

Each scenario includes:

- Available ingredient inventory
- Nutritional targets (calories, PFC ratio)
- Dietary restrictions and allergens

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

## Evaluation System

The evaluation system implements various reward functions.

### Scoring Formula and Evaluation Metrics

```python
total_score = (
    nutrition_score * 0.30 +
    constraint_score * 0.25 +
    inventory_score * 0.25 +
    quality_score * 0.20
)
```

Where:

- **nutrition_score**: PFC ratio and calorie accuracy (10% tolerance)
- **constraint_score**: Allergen compliance and dietary restrictions (zero tolerance for allergens)
- **inventory_score**: Utilization of available ingredients
- **quality_score**: Meal diversity, feasibility, and balance

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

TODO: The agent supports multiple LLM providers:

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

TODO: `config/prompts.yaml` is used only by the system prompt. The others are hard-coded so far.

## Future Enhancements

### Phase 2 Features:

- [ ] **Multi-Agent Architecture**: Separate planning, validation, and optimization agents
- [ ] **Best-of-N Selection**: Generate multiple plans and select best scoring
- [ ] **Async Parallel Processing**: Speed up API calls with concurrent requests
- [ ] **Recipe Database Integration**: Local cache for faster nutrition lookups
- [ ] **Meal Preference Learning**: Adapt to user preferences over time

### Advanced Evaluation:

- [ ] **LLM Judge Evaluation**: Taste and practicality scoring
- [ ] **Embedding Similarity**: Compare meal variety using embeddings

## Acknowledgments

- **FatSecret Platform**: Nutrition database API
- **OpenAI**: LLM capabilities that make this agent possible
