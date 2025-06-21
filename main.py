#!/usr/bin/env python3
import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from agents.base_agent import AgentConfig, ModelProvider
from agents.nutrition_planner import (
    DietaryConstraints,
    Inventory,
    NutritionPlannerAgent,
)

console = Console()
app = typer.Typer(help="Nutrition Agent - AI-powered meal planning assistant")


def load_sample_data(scenario_name: str) -> dict[str, Any]:
    """Load sample data for testing."""
    data_path = (
        Path(__file__).parent / "data" / "test_prompts" / f"{scenario_name}.json"
    )

    if not data_path.exists():
        console.print(f"[red]Sample data not found: {data_path}[/red]")
        return {}

    with open(data_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def create_interactive_inputs() -> tuple[Inventory, DietaryConstraints]:
    """Create inventory and constraints through interactive prompts."""
    console.print(
        Panel("🥘 [bold blue]Nutrition Agent - Interactive Setup[/bold blue]")
    )

    # Get inventory
    console.print("\n[yellow]Step 1: Enter your available ingredients[/yellow]")
    inventory_items = []

    while True:
        ingredient_name = Prompt.ask("Ingredient name (or 'done' to finish)")
        if ingredient_name.lower() == "done":
            break

        amount = typer.prompt(f"Amount of {ingredient_name} in grams", type=float)
        inventory_items.append(
            {"name": ingredient_name, "amount_g": amount, "unit": "g"}
        )

    inventory = Inventory(items=inventory_items)

    # Get dietary constraints
    console.print("\n[yellow]Step 2: Set your nutritional targets[/yellow]")

    daily_calories = typer.prompt("Daily calorie target", type=float, default=2000.0)

    console.print("PFC Ratio (should add up to 100%):")
    protein_pct = typer.prompt("Protein percentage", type=float, default=30.0)
    fat_pct = typer.prompt("Fat percentage", type=float, default=25.0)
    carbs_pct = typer.prompt("Carbohydrate percentage", type=float, default=45.0)

    # Normalize to 100%
    total_pct = protein_pct + fat_pct + carbs_pct
    if total_pct != 100:
        console.print(f"[yellow]Normalizing ratios (total was {total_pct}%)[/yellow]")
        protein_pct = protein_pct / total_pct * 100
        fat_pct = fat_pct / total_pct * 100
        carbs_pct = carbs_pct / total_pct * 100

    # Get restrictions
    allergens = []
    if Confirm.ask("Do you have any food allergies?"):
        allergen_input = Prompt.ask("List allergens (comma-separated)")
        allergens = [a.strip() for a in allergen_input.split(",")]

    dietary_restrictions = []
    if Confirm.ask("Do you have any dietary restrictions?"):
        restriction_input = Prompt.ask(
            "List restrictions (comma-separated, e.g., vegetarian, low-carb)"
        )
        dietary_restrictions = [r.strip() for r in restriction_input.split(",")]

    constraints = DietaryConstraints(
        daily_calories=daily_calories,
        pfc_ratio=(protein_pct, fat_pct, carbs_pct),
        allergens=allergens,
        dietary_restrictions=dietary_restrictions,
    )

    return inventory, constraints


@app.command()
def interactive() -> None:
    """Run the nutrition agent in interactive mode."""
    asyncio.run(run_interactive())


async def run_interactive() -> None:
    """Run interactive meal planning session."""
    # Get user inputs
    inventory, constraints = create_interactive_inputs()

    # Choose model
    console.print("\n[yellow]Step 3: Choose AI model[/yellow]")
    model_choice = Prompt.ask(
        "Select model",
        choices=["gpt-4o", "gpt-3.5-turbo", "claude-3-sonnet"],
        default="gpt-4o",
    )

    # Configure agent
    if model_choice.startswith("gpt"):
        config = AgentConfig(
            model_provider=ModelProvider.OPENAI,
            model_name=model_choice,
            temperature=0.7,
        )
    else:
        config = AgentConfig(
            model_provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.7,
        )

    # Get number of days
    days = typer.prompt("Number of days to plan", type=int, default=3)

    # Create and run agent
    agent = NutritionPlannerAgent(config)

    console.print("\n[green]Generating meal plan...[/green]")
    meal_plans = await agent.generate_meal_plan(inventory, constraints, days)

    # Display results
    agent.display_meal_plans(meal_plans)

    # Generate shopping list
    shopping_list = agent.generate_shopping_list(meal_plans)
    agent.display_shopping_list(shopping_list)

    # Ask if user wants to save results
    if Confirm.ask("\nSave meal plan to file?"):
        filename = Prompt.ask("Output filename", default="meal_plan.json")
        save_meal_plan(meal_plans, shopping_list, filename)


@app.command()
def sample(
    scenario: str = typer.Argument(..., help="Sample scenario name (t1, t2, t3)"),
    model: str = typer.Option("gpt-4o", help="Model to use"),
    days: int = typer.Option(3, help="Number of days to plan"),
) -> None:
    """Run with sample data."""
    asyncio.run(run_sample(scenario, model, days))


async def run_sample(scenario: str, model: str, days: int) -> None:
    """Run meal planning with sample data."""
    # Load sample data
    sample_data = load_sample_data(scenario)
    if not sample_data:
        return

    # Create inventory and constraints from sample data
    inventory = Inventory(items=sample_data["inventory"])
    constraints = DietaryConstraints(**sample_data["constraints"])

    # Configure agent
    if model.startswith("gpt"):
        config = AgentConfig(
            model_provider=ModelProvider.OPENAI, model_name=model, temperature=0.7
        )
    else:
        config = AgentConfig(
            model_provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.7,
        )

    # Display scenario info
    console.print(Panel(f"[bold blue]Running Scenario: {scenario.upper()}[/bold blue]"))
    console.print(f"Description: {sample_data.get('description', 'No description')}")

    # Create and run agent
    agent = NutritionPlannerAgent(config)

    console.print("\n[green]Generating meal plan...[/green]")
    meal_plans = await agent.generate_meal_plan(inventory, constraints, days)

    # Display results
    agent.display_meal_plans(meal_plans)

    # Generate shopping list
    shopping_list = agent.generate_shopping_list(meal_plans)
    agent.display_shopping_list(shopping_list)


@app.command()
def list_samples() -> None:
    """List available sample scenarios."""
    data_dir = Path(__file__).parent / "data" / "test_prompts"

    if not data_dir.exists():
        console.print("[red]No sample data directory found[/red]")
        return

    table = Table(title="Available Sample Scenarios")
    table.add_column("ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Special Notes", style="yellow")

    for json_file in data_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            table.add_row(
                json_file.stem,
                data.get("description", "No description"),
                data.get("notes", ""),
            )
        except Exception as e:
            table.add_row(json_file.stem, f"Error loading: {e}", "")

    console.print(table)


def save_meal_plan(
    meal_plans: list[Any], shopping_list: dict[str, Any], filename: str
) -> None:
    """Save meal plan to JSON file."""
    output_data = {
        "generated_at": "2025-06-20",
        "meal_plans": [
            {
                "day": plan.day,
                "meals": {
                    "breakfast": plan.breakfast,
                    "lunch": plan.lunch,
                    "dinner": plan.dinner,
                },
                "daily_nutrition": plan.daily_nutrition,
                "missing_ingredients": plan.missing_ingredients,
            }
            for plan in meal_plans
        ],
        "shopping_list": shopping_list,
    }

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"[green]Meal plan saved to {filename}[/green]")


if __name__ == "__main__":
    app()
