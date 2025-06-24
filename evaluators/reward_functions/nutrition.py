import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from agents.base_agent import AgentConfig
from agents.nutrition_planner import (
    DietaryConstraints,
    Inventory,
    MealPlan,
    NutritionPlannerAgent,
)
from evaluators.evaluator_manager import EvaluatorManager
from tools.nutrition_calculator import (
    MealNutrition,
    NutritionCalculator,
    NutritionTarget,
)

console = Console()


@dataclass
class EvaluationResult:
    scenario_id: str
    model_name: str
    score: float
    nutrition_score: float
    violations: list[str]
    execution_time: float
    nutrition_errors: dict[str, float]
    constraint_satisfaction_score: float = 0.0
    inventory_utilization_score: float = 0.0
    quality_scores: dict[str, float] = field(default_factory=dict)
    detailed_violations: dict[str, list[str]] = field(default_factory=dict)


class NutritionEvaluator:
    def __init__(self) -> None:
        self.tolerance_pct = 10.0  # 10% tolerance for nutrition targets
        self.evaluator_manager = EvaluatorManager()

    def calculate_nutrition_score(
        self, actual_nutrition: dict[str, Any], target_constraints: DietaryConstraints
    ) -> tuple[float, dict[str, float], list[str]]:
        """
        Calculate nutrition score based on actual vs target nutrition.

        Returns:
            (score, errors, violations)
        """
        # Create MealNutrition object from actual nutrition data
        actual = MealNutrition(
            meal_name="Daily total",
            calories=actual_nutrition.get("total_calories", 0),
            protein_g=actual_nutrition.get("total_protein_g", 0),
            fat_g=actual_nutrition.get("total_fat_g", 0),
            carbs_g=actual_nutrition.get("total_carbs_g", 0),
        )

        # Create NutritionTarget from constraints
        target = NutritionTarget(
            daily_calories=target_constraints.daily_calories,
            pfc_ratio=target_constraints.pfc_ratio,
        )

        # Use NutritionCalculator to calculate errors and check constraints
        calculator = NutritionCalculator()
        errors_raw = calculator.calculate_nutrition_error(actual, target)
        _, violations = calculator.check_nutrition_constraints(
            actual, target, self.tolerance_pct
        )

        # Convert error keys to match existing format
        errors = {
            "calories": errors_raw["calories_error"],
            "protein": errors_raw["protein_error"],
            "fat": errors_raw["fat_error"],
            "carbs": errors_raw["carbs_error"],
        }

        # Calculate overall nutrition score (0.0 to 1.0)
        max_error = max(errors.values()) if errors else 0
        if max_error <= self.tolerance_pct:
            nutrition_score = 1.0  # Perfect score
        else:
            # Linear decrease from 1.0 to 0.0 as error increases from 10% to 50%
            nutrition_score = max(
                0.0, 1.0 * (1 - (max_error - self.tolerance_pct) / 40)
            )

        return nutrition_score, errors, violations

    def calculate_constraint_satisfaction_score(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, list[str]]]:
        """
        Evaluate satisfaction of dietary constraints using the new evaluator system.

        Returns:
            (score, detailed_violations)
        """
        score, details = self.evaluator_manager.evaluate_constraint_satisfaction(
            meal_plans, constraints, inventory
        )
        violations = details.get("violations", {})
        return score, violations

    def calculate_inventory_utilization_score(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate how efficiently the available inventory is utilized using the new evaluator system.

        Returns:
            (score, detailed_metrics)
        """
        return self.evaluator_manager.evaluate_inventory_utilization(
            meal_plans, constraints, inventory
        )

    def calculate_quality_scores(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> dict[str, Any]:
        """
        Calculate all registered quality scores using the new evaluator system.

        Returns:
            Dictionary with quality scores and details
        """
        return self.evaluator_manager.evaluate_quality_scores(
            meal_plans, constraints, inventory
        )

    def calculate_overall_score(
        self,
        nutrition_score: float,
        constraint_satisfaction_score: float,
        inventory_utilization_score: float,
        quality_scores: dict[str, Any],
        has_critical_failure: bool = False,
    ) -> float:
        """
        Calculate overall score (0.0 to 1.0) using the new weighted evaluation system.
        """
        return self.evaluator_manager.calculate_overall_score(
            nutrition_score,
            constraint_satisfaction_score,
            inventory_utilization_score,
            quality_scores,
            has_critical_failure,
        )

    async def evaluate_scenario(
        self, scenario_path: Path, agent_config: AgentConfig, days: int = 3
    ) -> EvaluationResult:
        """
        Evaluate a single scenario.
        """
        import time

        start_time = time.time()

        # Load scenario data
        with open(scenario_path) as f:
            scenario_data = json.load(f)

        scenario_id = scenario_data["id"]
        inventory = Inventory(items=scenario_data["inventory"])
        constraints = DietaryConstraints(**scenario_data["constraints"])

        # Create agent and generate meal plan
        agent = NutritionPlannerAgent(agent_config)

        try:
            meal_plans = await agent.generate_meal_plan(inventory, constraints, days)

            # Extract nutrition info (simplified - in production would parse actual output)
            daily_nutrition = {
                "total_calories": sum(
                    plan.daily_nutrition["total_calories"] for plan in meal_plans
                )
                / len(meal_plans),
                "total_protein_g": sum(
                    plan.daily_nutrition["total_protein_g"] for plan in meal_plans
                )
                / len(meal_plans),
                "total_fat_g": sum(
                    plan.daily_nutrition["total_fat_g"] for plan in meal_plans
                )
                / len(meal_plans),
                "total_carbs_g": sum(
                    plan.daily_nutrition["total_carbs_g"] for plan in meal_plans
                )
                / len(meal_plans),
            }

            # Calculate all evaluation scores
            nutrition_score, nutrition_errors, nutrition_violations = (
                self.calculate_nutrition_score(daily_nutrition, constraints)
            )

            # Check for critical failures first
            has_critical_failure, critical_failure_msgs = (
                self.evaluator_manager.check_critical_failures(
                    meal_plans, constraints, inventory
                )
            )

            # Calculate new mandatory evaluations
            constraint_satisfaction_score, detailed_violations = (
                self.calculate_constraint_satisfaction_score(
                    meal_plans, constraints, inventory
                )
            )

            inventory_utilization_score, inventory_details = (
                self.calculate_inventory_utilization_score(
                    meal_plans, constraints, inventory
                )
            )

            # Calculate extensible quality scores
            quality_scores = self.calculate_quality_scores(
                meal_plans, constraints, inventory
            )

            # Calculate overall score using new system
            overall_score = self.calculate_overall_score(
                nutrition_score,
                constraint_satisfaction_score,
                inventory_utilization_score,
                quality_scores,
                has_critical_failure,
            )

            # Combine all violations
            all_violations = nutrition_violations.copy()
            all_violations.extend(critical_failure_msgs)
            for _, violation_list in detailed_violations.items():
                all_violations.extend(violation_list)

            execution_time = time.time() - start_time

            return EvaluationResult(
                scenario_id=scenario_id,
                model_name=agent_config.model_name,
                score=overall_score,
                nutrition_score=nutrition_score,
                violations=all_violations,
                execution_time=execution_time,
                nutrition_errors=nutrition_errors,
                constraint_satisfaction_score=constraint_satisfaction_score,
                inventory_utilization_score=inventory_utilization_score,
                quality_scores=quality_scores.get("individual_scores", {}),
                detailed_violations=detailed_violations,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            console.print(f"[red]Error evaluating scenario {scenario_id}: {e}[/red]")

            return EvaluationResult(
                scenario_id=scenario_id,
                model_name=agent_config.model_name,
                score=0.0,
                nutrition_score=0.0,
                violations=[f"Execution error: {str(e)}"],
                execution_time=execution_time,
                nutrition_errors={},
                constraint_satisfaction_score=0.0,
                inventory_utilization_score=0.0,
                quality_scores={},
                detailed_violations={},
            )

    async def evaluate_all_scenarios(
        self,
        scenarios_dir: Path,
        model_configs: list[tuple[str, AgentConfig]],
        days: int = 3,
    ) -> list[EvaluationResult]:
        """
        Evaluate all scenarios with all models.
        """
        results = []

        # Get all scenario files
        scenario_files = list(scenarios_dir.glob("*.json"))

        console.print(
            f"[blue]Evaluating {len(scenario_files)} scenarios with {len(model_configs)} models[/blue]"
        )

        for model_name, config in model_configs:
            console.print(f"\n[yellow]Testing model: {model_name}[/yellow]")

            for scenario_file in scenario_files:
                console.print(f"  Running scenario: {scenario_file.stem}")
                result = await self.evaluate_scenario(scenario_file, config, days)
                results.append(result)

        return results

    def display_results(self, results: list[EvaluationResult]) -> None:
        """
        Display evaluation results in a formatted table.
        """
        # Group by model
        models: dict[str, list[EvaluationResult]] = {}
        for result in results:
            if result.model_name not in models:
                models[result.model_name] = []
            models[result.model_name].append(result)

        # Create summary table with new evaluation metrics
        summary_table = Table(title="Model Performance Summary")
        summary_table.add_column("Model", style="cyan")
        summary_table.add_column("Total Score", justify="right")
        summary_table.add_column("Nutrition", justify="right")
        summary_table.add_column("Constraints", justify="right")
        summary_table.add_column("Inventory", justify="right")
        summary_table.add_column("Quality", justify="right")
        summary_table.add_column("Allergen Violations", justify="right")
        summary_table.add_column("Avg Time (s)", justify="right")

        for model_name, model_results in models.items():
            avg_score = np.mean([r.score for r in model_results])
            avg_nutrition = np.mean([r.nutrition_score for r in model_results])
            avg_constraints = np.mean(
                [r.constraint_satisfaction_score for r in model_results]
            )
            avg_inventory = np.mean(
                [r.inventory_utilization_score for r in model_results]
            )

            # Calculate average quality score
            quality_scores = []
            for result in model_results:
                if result.quality_scores:
                    quality_scores.append(np.mean(list(result.quality_scores.values())))
                else:
                    quality_scores.append(0.0)
            avg_quality = np.mean(quality_scores)

            avg_time = np.mean([r.execution_time for r in model_results])

            summary_table.add_row(
                model_name,
                f"{avg_score:.3f}",
                f"{avg_nutrition:.3f}",
                f"{avg_constraints:.3f}",
                f"{avg_inventory:.3f}",
                f"{avg_quality:.3f}",
                f"{avg_time:.1f}",
            )

        console.print(summary_table)

        # Detailed results by scenario
        for model_name, model_results in models.items():
            console.print(f"\n[bold]{model_name} - Detailed Results[/bold]")

            detail_table = Table()
            detail_table.add_column("Scenario", style="cyan")
            detail_table.add_column("Total", justify="right")
            detail_table.add_column("Nutrition", justify="right")
            detail_table.add_column("Constraints", justify="right")
            detail_table.add_column("Inventory", justify="right")
            detail_table.add_column("Quality", justify="right")
            detail_table.add_column("Key Violations", style="red")

            for result in model_results:
                # Calculate average quality score for this result
                if result.quality_scores:
                    quality_avg = np.mean(list(result.quality_scores.values()))
                else:
                    quality_avg = 0.0

                # Show key violations (first 2)
                violations_text = "; ".join(
                    result.violations[:2]
                )  # Show first 2 violations
                if len(result.violations) > 2:
                    violations_text += "..."

                detail_table.add_row(
                    result.scenario_id,
                    f"{result.score:.3f}",
                    f"{result.nutrition_score:.3f}",
                    f"{result.constraint_satisfaction_score:.3f}",
                    f"{result.inventory_utilization_score:.3f}",
                    f"{quality_avg:.3f}",
                    violations_text,
                )

            console.print(detail_table)

    def save_results(self, results: list[EvaluationResult], output_path: Path) -> None:
        """
        Save evaluation results to JSON file.
        """
        serializable_results = []
        for result in results:
            serializable_results.append(
                {
                    "scenario_id": result.scenario_id,
                    "model_name": result.model_name,
                    "score": result.score,
                    "nutrition_score": result.nutrition_score,
                    "violations": result.violations,
                    "execution_time": result.execution_time,
                    "nutrition_errors": result.nutrition_errors,
                    # New evaluation fields
                    "constraint_satisfaction_score": result.constraint_satisfaction_score,
                    "inventory_utilization_score": result.inventory_utilization_score,
                    "quality_scores": result.quality_scores,
                    "detailed_violations": result.detailed_violations,
                }
            )

        with open(output_path, "w") as f:
            json.dump(
                {"evaluation_timestamp": "2025-06-20", "results": serializable_results},
                f,
                indent=2,
            )

        console.print(f"[green]Results saved to {output_path}[/green]")
