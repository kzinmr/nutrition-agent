import json
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agents.nutrition_planner import MealPlan, DietaryConstraints, Inventory, NutritionPlannerAgent
from agents.base_agent import AgentConfig, ModelProvider

console = Console()


@dataclass
class EvaluationResult:
    scenario_id: str
    model_name: str
    score: float
    nutrition_score: float
    shopping_list_score: float
    allergen_violation: bool
    violations: List[str]
    execution_time: float
    nutrition_errors: Dict[str, float]
    jaccard_similarity: float


class NutritionEvaluator:
    def __init__(self):
        self.tolerance_pct = 10.0  # 10% tolerance for nutrition targets
    
    def calculate_nutrition_score(
        self,
        actual_nutrition: Dict[str, Any],
        target_constraints: DietaryConstraints
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Calculate nutrition score based on actual vs target nutrition.
        
        Returns:
            (score, errors, violations)
        """
        target_calories = target_constraints.daily_calories
        target_pfc = target_constraints.pfc_ratio
        
        # Calculate daily totals from meal plan
        actual_calories = actual_nutrition.get('total_calories', 0)
        actual_protein_g = actual_nutrition.get('total_protein_g', 0)
        actual_fat_g = actual_nutrition.get('total_fat_g', 0)
        actual_carbs_g = actual_nutrition.get('total_carbs_g', 0)
        
        # Calculate target macros in grams
        target_protein_g = (target_calories * target_pfc[0] / 100) / 4
        target_fat_g = (target_calories * target_pfc[1] / 100) / 9
        target_carbs_g = (target_calories * target_pfc[2] / 100) / 4
        
        # Calculate percentage errors
        errors = {}
        violations = []
        
        # Calorie error
        if target_calories > 0:
            errors['calories'] = abs(actual_calories - target_calories) / target_calories * 100
            if errors['calories'] > self.tolerance_pct:
                violations.append(f"Calories off by {errors['calories']:.1f}%")
        
        # Protein error
        if target_protein_g > 0:
            errors['protein'] = abs(actual_protein_g - target_protein_g) / target_protein_g * 100
            if errors['protein'] > self.tolerance_pct:
                violations.append(f"Protein off by {errors['protein']:.1f}%")
        
        # Fat error
        if target_fat_g > 0:
            errors['fat'] = abs(actual_fat_g - target_fat_g) / target_fat_g * 100
            if errors['fat'] > self.tolerance_pct:
                violations.append(f"Fat off by {errors['fat']:.1f}%")
        
        # Carbs error
        if target_carbs_g > 0:
            errors['carbs'] = abs(actual_carbs_g - target_carbs_g) / target_carbs_g * 100
            if errors['carbs'] > self.tolerance_pct:
                violations.append(f"Carbs off by {errors['carbs']:.1f}%")
        
        # Calculate overall nutrition score (0.0 to 0.5)
        max_error = max(errors.values()) if errors else 0
        if max_error <= self.tolerance_pct:
            nutrition_score = 0.5  # Perfect score
        else:
            # Linear decrease from 0.5 to 0.0 as error increases from 10% to 50%
            nutrition_score = max(0.0, 0.5 * (1 - (max_error - self.tolerance_pct) / 40))
        
        return nutrition_score, errors, violations
    
    def calculate_shopping_list_score(
        self,
        predicted_missing: List[str],
        ground_truth_missing: List[str]
    ) -> float:
        """
        Calculate Jaccard similarity between predicted and ground truth missing ingredients.
        """
        if not ground_truth_missing and not predicted_missing:
            return 1.0  # Both empty, perfect match
        
        # Convert to sets for easier comparison
        pred_set = set(predicted_missing)
        truth_set = set(ground_truth_missing)
        
        # Calculate Jaccard similarity
        intersection = len(pred_set.intersection(truth_set))
        union = len(pred_set.union(truth_set))
        
        if union == 0:
            return 1.0
        
        jaccard = intersection / union
        return jaccard * 0.5  # Shopping list contributes 0.5 to total score
    
    def check_allergen_violations(
        self,
        meal_plans: List[MealPlan],
        allergens: List[str]
    ) -> bool:
        """
        Check if any allergens appear in the meal plans.
        """
        if not allergens:
            return False
        
        # Simple keyword-based check
        # In production, this would use a proper allergen database
        for plan in meal_plans:
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                meal = getattr(plan, meal_type)
                meal_text = json.dumps(meal).lower()
                
                for allergen in allergens:
                    allergen_keywords = {
                        'dairy': ['milk', 'cheese', 'butter', 'cream', 'yogurt'],
                        'nuts': ['almond', 'peanut', 'walnut', 'cashew', 'pecan'],
                        'gluten': ['wheat', 'bread', 'pasta', 'flour', 'barley'],
                        'soy': ['soy', 'tofu', 'tempeh', 'miso'],
                        'shellfish': ['shrimp', 'crab', 'lobster', 'oyster'],
                        'fish': ['salmon', 'tuna', 'cod', 'trout']
                    }
                    
                    keywords = allergen_keywords.get(allergen.lower(), [allergen.lower()])
                    if any(keyword in meal_text for keyword in keywords):
                        return True
        
        return False
    
    def calculate_overall_score(
        self,
        nutrition_score: float,
        shopping_list_score: float,
        allergen_violation: bool
    ) -> float:
        """
        Calculate overall score (0.0 to 1.0).
        """
        if allergen_violation:
            return 0.0  # Immediate fail for allergen violations
        
        return nutrition_score + shopping_list_score
    
    async def evaluate_scenario(
        self,
        scenario_path: Path,
        agent_config: AgentConfig,
        days: int = 3
    ) -> EvaluationResult:
        """
        Evaluate a single scenario.
        """
        import time
        start_time = time.time()
        
        # Load scenario data
        with open(scenario_path, 'r') as f:
            scenario_data = json.load(f)
        
        scenario_id = scenario_data['id']
        inventory = Inventory(items=scenario_data['inventory'])
        constraints = DietaryConstraints(**scenario_data['constraints'])
        ground_truth = scenario_data.get('ground_truth', {})
        
        # Create agent and generate meal plan
        agent = NutritionPlannerAgent(agent_config)
        
        try:
            meal_plans = await agent.generate_meal_plan(inventory, constraints, days)
            
            # Extract nutrition info (simplified - in production would parse actual output)
            daily_nutrition = {
                'total_calories': sum(plan.daily_nutrition['total_calories'] for plan in meal_plans) / len(meal_plans),
                'total_protein_g': sum(plan.daily_nutrition['total_protein_g'] for plan in meal_plans) / len(meal_plans),
                'total_fat_g': sum(plan.daily_nutrition['total_fat_g'] for plan in meal_plans) / len(meal_plans),
                'total_carbs_g': sum(plan.daily_nutrition['total_carbs_g'] for plan in meal_plans) / len(meal_plans)
            }
            
            # Calculate scores
            nutrition_score, nutrition_errors, violations = self.calculate_nutrition_score(
                daily_nutrition, constraints
            )
            
            # Get predicted missing ingredients
            predicted_missing = []
            for plan in meal_plans:
                predicted_missing.extend(plan.missing_ingredients)
            predicted_missing = list(set(predicted_missing))  # Remove duplicates
            
            shopping_list_score = self.calculate_shopping_list_score(
                predicted_missing,
                ground_truth.get('expected_missing_ingredients', [])
            )
            
            jaccard_similarity = shopping_list_score / 0.5 if shopping_list_score > 0 else 0
            
            # Check allergen violations
            allergen_violation = self.check_allergen_violations(
                meal_plans, constraints.allergens or []
            )
            
            # Calculate overall score
            overall_score = self.calculate_overall_score(
                nutrition_score, shopping_list_score, allergen_violation
            )
            
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                scenario_id=scenario_id,
                model_name=agent_config.model_name,
                score=overall_score,
                nutrition_score=nutrition_score,
                shopping_list_score=shopping_list_score,
                allergen_violation=allergen_violation,
                violations=violations,
                execution_time=execution_time,
                nutrition_errors=nutrition_errors,
                jaccard_similarity=jaccard_similarity
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            console.print(f"[red]Error evaluating scenario {scenario_id}: {e}[/red]")
            
            return EvaluationResult(
                scenario_id=scenario_id,
                model_name=agent_config.model_name,
                score=0.0,
                nutrition_score=0.0,
                shopping_list_score=0.0,
                allergen_violation=False,
                violations=[f"Execution error: {str(e)}"],
                execution_time=execution_time,
                nutrition_errors={},
                jaccard_similarity=0.0
            )
    
    async def evaluate_all_scenarios(
        self,
        scenarios_dir: Path,
        model_configs: List[Tuple[str, AgentConfig]],
        days: int = 3
    ) -> List[EvaluationResult]:
        """
        Evaluate all scenarios with all models.
        """
        results = []
        
        # Get all scenario files
        scenario_files = list(scenarios_dir.glob("*.json"))
        
        console.print(f"[blue]Evaluating {len(scenario_files)} scenarios with {len(model_configs)} models[/blue]")
        
        for model_name, config in model_configs:
            console.print(f"\n[yellow]Testing model: {model_name}[/yellow]")
            
            for scenario_file in scenario_files:
                console.print(f"  Running scenario: {scenario_file.stem}")
                result = await self.evaluate_scenario(scenario_file, config, days)
                results.append(result)
        
        return results
    
    def display_results(self, results: List[EvaluationResult]):
        """
        Display evaluation results in a formatted table.
        """
        # Group by model
        models = {}
        for result in results:
            if result.model_name not in models:
                models[result.model_name] = []
            models[result.model_name].append(result)
        
        # Create summary table
        summary_table = Table(title="Model Performance Summary")
        summary_table.add_column("Model", style="cyan")
        summary_table.add_column("Avg Score", justify="right")
        summary_table.add_column("Nutrition Score", justify="right")
        summary_table.add_column("Shopping Score", justify="right")
        summary_table.add_column("Allergen Violations", justify="right")
        summary_table.add_column("Avg Time (s)", justify="right")
        
        for model_name, model_results in models.items():
            avg_score = np.mean([r.score for r in model_results])
            avg_nutrition = np.mean([r.nutrition_score for r in model_results])
            avg_shopping = np.mean([r.shopping_list_score for r in model_results])
            allergen_violations = sum([r.allergen_violation for r in model_results])
            avg_time = np.mean([r.execution_time for r in model_results])
            
            summary_table.add_row(
                model_name,
                f"{avg_score:.3f}",
                f"{avg_nutrition:.3f}",
                f"{avg_shopping:.3f}",
                str(allergen_violations),
                f"{avg_time:.1f}"
            )
        
        console.print(summary_table)
        
        # Detailed results by scenario
        for model_name, model_results in models.items():
            console.print(f"\n[bold]{model_name} - Detailed Results[/bold]")
            
            detail_table = Table()
            detail_table.add_column("Scenario", style="cyan")
            detail_table.add_column("Score", justify="right")
            detail_table.add_column("Nutrition", justify="right")
            detail_table.add_column("Shopping", justify="right")
            detail_table.add_column("Violations", style="red")
            
            for result in model_results:
                violations_text = "; ".join(result.violations[:2])  # Show first 2 violations
                if len(result.violations) > 2:
                    violations_text += "..."
                
                detail_table.add_row(
                    result.scenario_id,
                    f"{result.score:.3f}",
                    f"{result.nutrition_score:.3f}",
                    f"{result.shopping_list_score:.3f}",
                    violations_text
                )
            
            console.print(detail_table)
    
    def save_results(self, results: List[EvaluationResult], output_path: Path):
        """
        Save evaluation results to JSON file.
        """
        serializable_results = []
        for result in results:
            serializable_results.append({
                "scenario_id": result.scenario_id,
                "model_name": result.model_name,
                "score": result.score,
                "nutrition_score": result.nutrition_score,
                "shopping_list_score": result.shopping_list_score,
                "allergen_violation": result.allergen_violation,
                "violations": result.violations,
                "execution_time": result.execution_time,
                "nutrition_errors": result.nutrition_errors,
                "jaccard_similarity": result.jaccard_similarity
            })
        
        with open(output_path, 'w') as f:
            json.dump({
                "evaluation_timestamp": "2025-06-20",
                "results": serializable_results
            }, f, indent=2)
        
        console.print(f"[green]Results saved to {output_path}[/green]")