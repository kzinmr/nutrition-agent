#!/usr/bin/env python3
"""
Evaluation script for the Nutrition Agent.

This script runs the nutrition agent on test scenarios and evaluates performance
using the reward function defined in first_assignment.md.
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from agents.base_agent import AgentConfig, ModelProvider
from evaluators.nutrition_evaluator import NutritionEvaluator

console = Console()
app = typer.Typer(help="Nutrition Agent Evaluation Tool")


def get_model_configs(models: list[str]) -> list[tuple[str, AgentConfig]]:
    """Convert model names to AgentConfig objects."""
    configs = []

    for model in models:
        if model.startswith("gpt"):
            config = AgentConfig(
                model_provider=ModelProvider.OPENAI,
                model_name=model,
                temperature=0.7,
                max_tokens=4000,
            )
        elif model.startswith("claude"):
            config = AgentConfig(
                model_provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=4000,
            )
        else:
            console.print(f"[red]Unknown model: {model}[/red]")
            continue

        configs.append((model, config))

    return configs


@app.command()
def run(
    scenarios_dir: str = typer.Option(
        "data/test_prompts", help="Directory containing test scenario JSON files"
    ),
    models: list[str] = typer.Option(
        ["gpt-4o", "gpt-3.5-turbo"], help="Models to evaluate"
    ),
    days: int = typer.Option(3, help="Number of days to plan"),
    output: str = typer.Option(
        "evaluation_results.json", help="Output file for results"
    ),
) -> None:
    """Run evaluation on all scenarios with specified models."""
    asyncio.run(run_evaluation(scenarios_dir, models, days, output))


async def run_evaluation(
    scenarios_dir: str, models: list[str], days: int, output: str
) -> None:
    """Main evaluation function."""
    scenarios_path = Path(scenarios_dir)
    output_path = Path(output)

    if not scenarios_path.exists():
        console.print(f"[red]Scenarios directory not found: {scenarios_path}[/red]")
        return

    # Get model configurations
    model_configs = get_model_configs(models)
    if not model_configs:
        console.print("[red]No valid models specified[/red]")
        return

    # Create evaluator
    evaluator = NutritionEvaluator()

    # Run evaluation
    console.print("[bold blue]Starting Nutrition Agent Evaluation[/bold blue]")
    results = await evaluator.evaluate_all_scenarios(
        scenarios_path, model_configs, days
    )

    # Display results
    evaluator.display_results(results)

    # Save results
    evaluator.save_results(results, output_path)


@app.command()
def single(
    scenario: str = typer.Argument(help="Scenario file (e.g., t1.json)"),
    model: str = typer.Option("gpt-4o", help="Model to use"),
    days: int = typer.Option(3, help="Number of days to plan"),
) -> None:
    """Evaluate a single scenario with one model."""
    asyncio.run(run_single_evaluation(scenario, model, days))


async def run_single_evaluation(scenario: str, model: str, days: int) -> None:
    """Run evaluation on a single scenario."""
    scenario_path = Path("data/test_prompts") / scenario

    if not scenario_path.exists():
        console.print(f"[red]Scenario file not found: {scenario_path}[/red]")
        return

    # Get model configuration
    model_configs = get_model_configs([model])
    if not model_configs:
        console.print(f"[red]Invalid model: {model}[/red]")
        return

    # Create evaluator
    evaluator = NutritionEvaluator()

    # Run evaluation
    console.print(f"[bold blue]Evaluating scenario {scenario} with {model}[/bold blue]")
    result = await evaluator.evaluate_scenario(scenario_path, model_configs[0][1], days)

    # Display result
    console.print(f"\n[bold green]Results for {scenario}:[/bold green]")
    console.print(f"Overall Score: {result.score:.3f}/1.0")
    console.print(f"Nutrition Score: {result.nutrition_score:.3f}/0.5")
    console.print(f"Shopping List Score: {result.shopping_list_score:.3f}/0.5")
    console.print(f"Execution Time: {result.execution_time:.1f}s")

    if result.allergen_violation:
        console.print("[red]ALLERGEN VIOLATION DETECTED![/red]")

    if result.violations:
        console.print("\n[yellow]Violations:[/yellow]")
        for violation in result.violations:
            console.print(f"  • {violation}")

    if result.nutrition_errors:
        console.print("\n[cyan]Nutrition Errors:[/cyan]")
        for macro, error in result.nutrition_errors.items():
            console.print(f"  • {macro}: {error:.1f}%")


@app.command()
def compare(
    models: list[str] = typer.Option(
        ["gpt-4o", "gpt-3.5-turbo", "claude-3-sonnet"], help="Models to compare"
    ),
    output: str = typer.Option(
        "model_comparison.json", help="Output file for comparison"
    ),
) -> None:
    """Compare multiple models across all scenarios."""
    asyncio.run(run_model_comparison(models, output))


async def run_model_comparison(models: list[str], output: str) -> None:
    """Run comprehensive model comparison."""
    scenarios_path = Path("data/test_prompts")
    output_path = Path(output)

    if not scenarios_path.exists():
        console.print(f"[red]Scenarios directory not found: {scenarios_path}[/red]")
        return

    # Get model configurations
    model_configs = get_model_configs(models)
    if not model_configs:
        console.print("[red]No valid models specified[/red]")
        return

    # Create evaluator
    evaluator = NutritionEvaluator()

    # Run evaluation
    console.print("[bold blue]Running Model Comparison[/bold blue]")
    results = await evaluator.evaluate_all_scenarios(
        scenarios_path, model_configs, days=3
    )

    # Display results
    evaluator.display_results(results)

    # Save results
    evaluator.save_results(results, output_path)

    # Additional analysis
    console.print("\n[bold yellow]Model Ranking:[/bold yellow]")

    # Group by model and calculate average scores
    model_scores: dict[str, list[float]] = {}
    for result in results:
        if result.model_name not in model_scores:
            model_scores[result.model_name] = []
        model_scores[result.model_name].append(result.score)

    # Sort by average score
    sorted_models = sorted(
        model_scores.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True
    )

    for rank, (model, scores) in enumerate(sorted_models, 1):
        avg_score = sum(scores) / len(scores)
        console.print(
            f"{rank}. {model}: {avg_score:.3f} (±{max(scores) - min(scores):.3f})"
        )


@app.command()
def validate() -> None:
    """Validate evaluation setup and test scenarios."""
    scenarios_path = Path("data/test_prompts")

    if not scenarios_path.exists():
        console.print(f"[red]Scenarios directory not found: {scenarios_path}[/red]")
        return

    console.print("[bold blue]Validating Test Scenarios[/bold blue]")

    scenario_files = list(scenarios_path.glob("*.json"))
    if not scenario_files:
        console.print("[red]No scenario files found[/red]")
        return

    import json

    valid_scenarios = 0

    for scenario_file in scenario_files:
        try:
            with open(scenario_file) as f:
                data = json.load(f)

            # Basic validation
            required_keys = ["id", "inventory", "constraints"]
            missing_keys = [key for key in required_keys if key not in data]

            if missing_keys:
                console.print(
                    f"[red]❌ {scenario_file.name}: Missing keys: {missing_keys}[/red]"
                )
            else:
                console.print(f"[green]✅ {scenario_file.name}: Valid[/green]")
                valid_scenarios += 1

        except Exception as e:
            console.print(f"[red]❌ {scenario_file.name}: Error loading: {e}[/red]")

    console.print(
        f"\n[bold]Summary: {valid_scenarios}/{len(scenario_files)} scenarios are valid[/bold]"
    )


if __name__ == "__main__":
    app()
