"""
Evaluator manager for coordinating multiple evaluation metrics.
"""

from typing import Any

from rich.console import Console

from agents.nutrition_planner import DietaryConstraints, Inventory, MealPlan
from evaluators.reward_functions.base import (
    MandatoryEvaluator,
    QualityEvaluator,
)
from evaluators.reward_functions.constraint import (
    ConstraintSatisfactionEvaluator,
)
from evaluators.reward_functions.inventory import (
    InventoryUtilizationEvaluator,
)
from evaluators.reward_functions.quality import (
    DiversityEvaluator,
    FeasibilityEvaluator,
    NutritionalBalanceEvaluator,
)

console = Console()


class EvaluatorManager:
    """Manages and coordinates multiple evaluation metrics."""

    def __init__(self):
        # Score weights (must sum to 1.0)
        self.score_weights = {
            "nutrition": 0.30,  # Nutrition balance (external)
            "constraint_satisfaction": 0.25,  # Mandatory: allergens, restrictions
            "inventory_utilization": 0.25,  # Mandatory: inventory usage
            "quality": 0.20,  # Extensible quality metrics
        }

        # Initialize evaluators
        self.mandatory_evaluators: dict[str, MandatoryEvaluator] = {}
        self.quality_evaluators: dict[str, QualityEvaluator] = {}

        self._register_default_evaluators()

    def _register_default_evaluators(self):
        """Register default evaluators."""
        # Mandatory evaluators
        self.register_mandatory_evaluator(ConstraintSatisfactionEvaluator())
        self.register_mandatory_evaluator(InventoryUtilizationEvaluator())

        # Quality evaluators
        self.register_quality_evaluator(DiversityEvaluator())
        self.register_quality_evaluator(FeasibilityEvaluator())
        self.register_quality_evaluator(NutritionalBalanceEvaluator())

    def register_mandatory_evaluator(self, evaluator: MandatoryEvaluator):
        """Register a mandatory evaluator."""
        self.mandatory_evaluators[evaluator.name] = evaluator

    def register_quality_evaluator(self, evaluator: QualityEvaluator):
        """Register a quality evaluator."""
        self.quality_evaluators[evaluator.name] = evaluator

    def unregister_evaluator(self, name: str):
        """Remove an evaluator by name."""
        if name in self.mandatory_evaluators:
            del self.mandatory_evaluators[name]
        elif name in self.quality_evaluators:
            del self.quality_evaluators[name]

    def evaluate_constraint_satisfaction(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate constraint satisfaction (mandatory)."""
        evaluator = self.mandatory_evaluators.get("constraint_satisfaction")
        if not evaluator:
            return 0.0, {"error": "Constraint satisfaction evaluator not found"}

        return evaluator.evaluate(meal_plans, constraints, inventory)

    def evaluate_inventory_utilization(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate inventory utilization (mandatory)."""
        evaluator = self.mandatory_evaluators.get("inventory_utilization")
        if not evaluator:
            return 0.0, {"error": "Inventory utilization evaluator not found"}

        return evaluator.evaluate(meal_plans, constraints, inventory)

    def evaluate_quality_scores(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> dict[str, Any]:
        """Evaluate all quality metrics."""
        quality_scores = {}
        quality_details = {}

        for name, evaluator in self.quality_evaluators.items():
            try:
                score, details = evaluator.evaluate(meal_plans, constraints, inventory)
                quality_scores[name] = score
                quality_details[name] = details
            except Exception as e:
                console.print(f"[red]Error in quality evaluator '{name}': {e}[/red]")
                quality_scores[name] = 0.0
                quality_details[name] = {"error": str(e)}

        # Calculate weighted average
        if quality_scores:
            total_weight = sum(
                evaluator.weight for evaluator in self.quality_evaluators.values()
            )
            weighted_score = (
                sum(
                    score * self.quality_evaluators[name].weight
                    for name, score in quality_scores.items()
                )
                / total_weight
                if total_weight > 0
                else 0.0
            )
        else:
            weighted_score = 0.0

        return {
            "total_score": weighted_score,
            "individual_scores": quality_scores,
            "details": quality_details,
        }

    def check_critical_failures(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[bool, list[str]]:
        """Check for critical failures that should result in score = 0."""
        critical_failures = []
        has_critical_failure = False

        for name, evaluator in self.mandatory_evaluators.items():
            if evaluator.is_critical_failure(meal_plans, constraints, inventory):
                has_critical_failure = True
                critical_failures.append(f"Critical failure in {name}")

        return has_critical_failure, critical_failures

    def calculate_overall_score(
        self,
        nutrition_score: float,
        constraint_satisfaction_score: float,
        inventory_utilization_score: float,
        quality_scores: dict[str, Any],
        has_critical_failure: bool = False,
    ) -> float:
        """Calculate overall weighted score."""
        if has_critical_failure:
            return 0.0

        # Calculate weighted total score
        total_score = (
            nutrition_score * self.score_weights["nutrition"]
            + constraint_satisfaction_score
            * self.score_weights["constraint_satisfaction"]
            + inventory_utilization_score * self.score_weights["inventory_utilization"]
            + quality_scores.get("total_score", 0.0) * self.score_weights["quality"]
        )

        return min(1.0, max(0.0, total_score))

    def get_evaluator_info(self) -> dict[str, Any]:
        """Get information about registered evaluators."""
        info = {
            "score_weights": self.score_weights,
            "mandatory_evaluators": {
                name: {
                    "name": evaluator.name,
                    "description": evaluator.description,
                    "weight": evaluator.weight,
                }
                for name, evaluator in self.mandatory_evaluators.items()
            },
            "quality_evaluators": {
                name: {
                    "name": evaluator.name,
                    "description": evaluator.description,
                    "weight": evaluator.weight,
                }
                for name, evaluator in self.quality_evaluators.items()
            },
        }
        return info

    def update_score_weights(self, new_weights: dict[str, float]):
        """Update scoring weights (must sum to 1.0)."""
        if abs(sum(new_weights.values()) - 1.0) > 0.001:
            raise ValueError("Score weights must sum to 1.0")

        self.score_weights.update(new_weights)

    def update_evaluator_weight(self, evaluator_name: str, new_weight: float):
        """Update the weight of a specific evaluator."""
        if evaluator_name in self.mandatory_evaluators:
            self.mandatory_evaluators[evaluator_name].weight = new_weight
        elif evaluator_name in self.quality_evaluators:
            self.quality_evaluators[evaluator_name].weight = new_weight
        else:
            raise ValueError(f"Evaluator '{evaluator_name}' not found")
