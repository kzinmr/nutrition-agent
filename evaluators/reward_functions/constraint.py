"""
Constraint satisfaction evaluator for nutrition meal plans.
"""

import json
from typing import Any

import numpy as np

from agents.nutrition_planner import DietaryConstraints, Inventory, MealPlan
from evaluators.reward_functions.base import MandatoryEvaluator


class ConstraintSatisfactionEvaluator(MandatoryEvaluator):
    """Evaluates satisfaction of dietary constraints (allergens, dietary restrictions)."""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        self.allergen_keywords = {
            "dairy": [
                "milk",
                "cheese",
                "butter",
                "cream",
                "yogurt",
            ],
            "nuts": [
                "almond",
                "peanut",
                "walnut",
                "cashew",
                "pecan",
            ],
            "gluten": [
                "wheat",
                "bread",
                "pasta",
                "flour",
                "barley",
            ],
            "soy": [
                "soy",
                "tofu",
                "tempeh",
                "miso",
            ],
            "shellfish": [
                "shrimp",
                "crab",
                "lobster",
                "oyster",
            ],
            "fish": [
                "salmon",
                "tuna",
                "cod",
                "trout",
            ],
        }

    @property
    def name(self) -> str:
        return "constraint_satisfaction"

    @property
    def description(self) -> str:
        return "Evaluates adherence to dietary constraints including allergens and restrictions"

    def evaluate(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate satisfaction of dietary constraints.

        Returns:
            (score, detailed_violations)
        """
        violations = {
            "allergen_violations": [],
            "dietary_restriction_violations": [],
            "meal_distribution_violations": [],
        }

        component_scores = {}

        # 1. Allergen compliance (CRITICAL - zero tolerance)
        allergen_violation = self._check_allergen_violations(
            meal_plans, constraints.allergens or []
        )
        if allergen_violation:
            violations["allergen_violations"].append("Allergen detected in meal plan")
            component_scores["allergen"] = 0.0
        else:
            component_scores["allergen"] = 1.0

        # 2. Dietary restriction compliance
        restriction_score = self._evaluate_dietary_restrictions(meal_plans, constraints)
        component_scores["dietary_restrictions"] = restriction_score["score"]
        violations["dietary_restriction_violations"] = restriction_score["violations"]

        # 3. Meal distribution balance (breakfast 25%, lunch 35%, dinner 40%)
        distribution_score = self._evaluate_meal_distribution(meal_plans)
        component_scores["meal_distribution"] = distribution_score["score"]
        violations["meal_distribution_violations"] = distribution_score["violations"]

        # Calculate weighted score (allergen compliance is critical)
        if component_scores["allergen"] == 0.0:
            total_score = 0.0  # Immediate fail for allergen violations
        else:
            total_score = (
                component_scores["allergen"] * 0.5
                + component_scores["dietary_restrictions"] * 0.3
                + component_scores["meal_distribution"] * 0.2
            )

        details = {
            "component_scores": component_scores,
            "violations": violations,
            "total_violations": sum(len(v) for v in violations.values()),
        }

        return total_score, details

    def is_critical_failure(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> bool:
        """Check if there are allergen violations (critical failure)."""
        return self._check_allergen_violations(meal_plans, constraints.allergens or [])

    def _check_allergen_violations(
        self, meal_plans: list[MealPlan], allergens: list[str]
    ) -> bool:
        """Check if any allergens appear in the meal plans."""
        if not allergens:
            return False

        for plan in meal_plans:
            for meal_type in ["breakfast", "lunch", "dinner"]:
                meal = getattr(plan, meal_type)
                meal_text = json.dumps(meal).lower()

                for allergen in allergens:
                    keywords = self.allergen_keywords.get(
                        allergen.lower(), [allergen.lower()]
                    )
                    if any(keyword in meal_text for keyword in keywords):
                        return True

        return False

    def _evaluate_dietary_restrictions(
        self, meal_plans: list[MealPlan], constraints: DietaryConstraints
    ) -> dict[str, Any]:
        """Evaluate adherence to dietary restrictions."""
        violations = []

        for restriction in constraints.dietary_restrictions or []:
            restriction_lower = restriction.lower()

            for plan in meal_plans:
                for meal_type, meal in [
                    ("breakfast", plan.breakfast),
                    ("lunch", plan.lunch),
                    ("dinner", plan.dinner),
                ]:
                    meal_text = json.dumps(meal).lower()

                    if restriction_lower == "vegetarian":
                        meat_keywords = [
                            "meat",
                            "beef",
                            "pork",
                            "chicken",
                            "fish",
                            "seafood",
                        ]
                        if any(keyword in meal_text for keyword in meat_keywords):
                            violations.append(
                                f"Vegetarian violation in {meal_type} on day {plan.day}"
                            )

                    elif restriction_lower == "vegan":
                        animal_keywords = [
                            "meat",
                            "dairy",
                            "egg",
                            "milk",
                            "cheese",
                            "butter",
                            "yogurt",
                        ]
                        if any(keyword in meal_text for keyword in animal_keywords):
                            violations.append(
                                f"Vegan violation in {meal_type} on day {plan.day}"
                            )

                    elif restriction_lower == "low-carb":
                        high_carb_keywords = [
                            "bread",
                            "rice",
                            "pasta",
                            "potato",
                            "noodle",
                        ]
                        if any(keyword in meal_text for keyword in high_carb_keywords):
                            violations.append(
                                f"Low-carb violation in {meal_type} on day {plan.day}"
                            )

        score = 1.0 if not violations else max(0.0, 1.0 - len(violations) * 0.2)
        return {"score": score, "violations": violations}

    def _evaluate_meal_distribution(self, meal_plans: list[MealPlan]) -> dict[str, Any]:
        """Evaluate if meals are properly distributed (breakfast 25%, lunch 35%, dinner 40%)."""
        violations = []
        distribution_scores = []

        ideal_distribution = [0.25, 0.35, 0.40]  # breakfast, lunch, dinner

        for plan in meal_plans:
            breakfast_cal = plan.breakfast.get("calories", 0)
            lunch_cal = plan.lunch.get("calories", 0)
            dinner_cal = plan.dinner.get("calories", 0)
            total_cal = breakfast_cal + lunch_cal + dinner_cal

            if total_cal > 0:
                actual_distribution = [
                    breakfast_cal / total_cal,
                    lunch_cal / total_cal,
                    dinner_cal / total_cal,
                ]
                distribution_error = sum(
                    abs(a - i)
                    for a, i in zip(
                        actual_distribution, ideal_distribution, strict=True
                    )
                )

                if distribution_error > 0.2:  # Allow 20% total deviation
                    violations.append(
                        f"Poor meal distribution on day {plan.day}: "
                        f"{[f'{d:.1%}' for d in actual_distribution]}"
                    )

                distribution_scores.append(1.0 - min(distribution_error, 1.0))
            else:
                distribution_scores.append(0.0)

        avg_score = float(np.mean(distribution_scores)) if distribution_scores else 0.0
        return {"score": avg_score, "violations": violations}
