"""
Inventory utilization evaluator for nutrition meal plans.
"""

from typing import Any

from agents.nutrition_planner import DietaryConstraints, Inventory, MealPlan
from evaluators.reward_functions.base import MandatoryEvaluator


class InventoryUtilizationEvaluator(MandatoryEvaluator):
    """Evaluates how efficiently the available inventory is utilized."""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        self.basic_ingredients = {
            "salt",
            "sugar",
            "oil",
            "soy sauce",
            "vinegar",
            "pepper",
            "butter",
            "milk",
            "eggs",
        }
        self.specialty_ingredients = {
            "truffle",
            "caviar",
            "wagyu",
            "exotic spices",
            "rare herbs",
        }

    @property
    def name(self) -> str:
        return "inventory_utilization"

    @property
    def description(self) -> str:
        return "Evaluates efficient use of available inventory and reasonableness of missing ingredients"

    def evaluate(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate how efficiently the available inventory is utilized.

        Returns:
            (score, detailed_metrics)
        """
        # Extract available ingredients
        available_items = {item["name"].lower() for item in inventory.items}
        used_items = set()
        all_ingredients = []
        missing_ingredients = []

        # Collect all ingredients used and missing
        for plan in meal_plans:
            for meal in [plan.breakfast, plan.lunch, plan.dinner]:
                ingredients = meal.get("ingredients", [])
                all_ingredients.extend(ingredients)

                # Check which available ingredients are used
                for ingredient in ingredients:
                    ingredient_lower = ingredient.lower()
                    for available_item in available_items:
                        if (
                            available_item in ingredient_lower
                            or ingredient_lower in available_item
                        ):
                            used_items.add(available_item)

            missing_ingredients.extend(plan.missing_ingredients)

        # Calculate utilization metrics
        inventory_utilization_rate = (
            len(used_items) / len(available_items) if available_items else 1.0
        )

        # Evaluate missing ingredients reasonableness
        missing_score = self._evaluate_missing_ingredients_quality(missing_ingredients)

        # Calculate ingredient efficiency (avoid excessive ingredient count)
        avg_ingredients_per_meal = len(set(all_ingredients)) / (len(meal_plans) * 3)
        efficiency_score = 1.0 if 3 <= avg_ingredients_per_meal <= 8 else 0.7

        # Combined score
        total_score = (
            inventory_utilization_rate * 0.5
            + missing_score * 0.3
            + efficiency_score * 0.2
        )

        details = {
            "inventory_utilization_rate": inventory_utilization_rate,
            "missing_ingredients_quality": missing_score,
            "ingredient_efficiency": efficiency_score,
            "used_items_count": len(used_items),
            "available_items_count": len(available_items),
            "avg_ingredients_per_meal": avg_ingredients_per_meal,
            "missing_ingredients": missing_ingredients,
            "used_items": list(used_items),
        }

        return total_score, details

    def is_critical_failure(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> bool:
        """Inventory utilization failures are not critical."""
        return False

    def _evaluate_missing_ingredients_quality(
        self, missing_ingredients: list[str]
    ) -> float:
        """Evaluate the reasonableness of missing ingredients."""
        if not missing_ingredients:
            return 1.0

        basic_count = 0
        specialty_count = 0

        for ingredient in missing_ingredients:
            ingredient_lower = ingredient.lower()

            if any(basic in ingredient_lower for basic in self.basic_ingredients):
                basic_count += 1
            elif any(
                specialty in ingredient_lower
                for specialty in self.specialty_ingredients
            ):
                specialty_count += 1

        total_missing = len(missing_ingredients)
        basic_ratio = basic_count / total_missing if total_missing > 0 else 0
        specialty_penalty = specialty_count * 0.2

        # Score higher for basic ingredients, penalize for specialty ingredients
        score = max(0.0, basic_ratio - specialty_penalty)
        return min(1.0, score)
