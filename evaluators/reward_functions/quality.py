"""
Quality evaluators for nutrition meal plans.
"""

from typing import Any

import numpy as np

from agents.nutrition_planner import DietaryConstraints, Inventory, MealPlan
from evaluators.reward_functions.base import QualityEvaluator


class DiversityEvaluator(QualityEvaluator):
    """Evaluates ingredient and cooking method diversity."""

    @property
    def name(self) -> str:
        return "diversity"

    @property
    def description(self) -> str:
        return (
            "Evaluates diversity of ingredients, cooking methods, and protein sources"
        )

    def evaluate(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate ingredient and cooking method diversity."""
        all_ingredients = []
        cooking_methods = []
        protein_sources = []

        for plan in meal_plans:
            for meal in [plan.breakfast, plan.lunch, plan.dinner]:
                ingredients = meal.get("ingredients", [])
                all_ingredients.extend(ingredients)

                # Extract cooking methods from instructions
                instructions = meal.get("cooking_instructions", "").lower()
                if any(
                    word in instructions
                    for word in [
                        "grill",
                        "roast",
                        "bake",
                    ]
                ):
                    cooking_methods.append("grill/roast")
                if any(
                    word in instructions
                    for word in [
                        "simmer",
                        "boil",
                    ]
                ):
                    cooking_methods.append("simmer/boil")
                if any(
                    word in instructions
                    for word in [
                        "stir-fry",
                        "saute",
                    ]
                ):
                    cooking_methods.append("stir-fry")
                if any(
                    word in instructions
                    for word in [
                        "steam",
                    ]
                ):
                    cooking_methods.append("steam")
                if any(
                    word in instructions
                    for word in [
                        "fry",
                    ]
                ):
                    cooking_methods.append("deep-fry")

                # Identify protein sources
                for ingredient in ingredients:
                    ing_lower = ingredient.lower()
                    if any(
                        protein in ing_lower
                        for protein in [
                            "meat",
                            "fish",
                            "tofu",
                            "egg",
                            "chicken",
                            "beef",
                            "pork",
                        ]
                    ):
                        protein_sources.append(ingredient)

        # Calculate diversity scores
        ingredient_diversity = (
            len(set(all_ingredients)) / len(all_ingredients) if all_ingredients else 0
        )
        method_diversity = (
            len(set(cooking_methods)) / (len(meal_plans) * 3) if meal_plans else 0
        )
        protein_diversity = (
            len(set(protein_sources)) / len(protein_sources) if protein_sources else 0
        )

        overall_score = (
            ingredient_diversity * 0.5
            + method_diversity * 0.3
            + protein_diversity * 0.2
        )

        details = {
            "ingredient_diversity": ingredient_diversity,
            "cooking_method_diversity": method_diversity,
            "protein_source_diversity": protein_diversity,
            "unique_ingredients": len(set(all_ingredients)),
            "unique_cooking_methods": len(set(cooking_methods)),
            "unique_protein_sources": len(set(protein_sources)),
            "total_ingredients": len(all_ingredients),
            "cooking_methods_used": list(set(cooking_methods)),
            "protein_sources_used": list(set(protein_sources)),
        }

        return overall_score, details


class FeasibilityEvaluator(QualityEvaluator):
    """Evaluates cooking feasibility and practicality."""

    @property
    def name(self) -> str:
        return "feasibility"

    @property
    def description(self) -> str:
        return "Evaluates cooking complexity and practicality of meal preparation"

    def evaluate(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate cooking feasibility and practicality."""
        feasibility_scores = []
        complexity_details = []
        ingredient_count_details = []

        for plan in meal_plans:
            day_scores = []
            for meal_type, meal in [
                ("breakfast", plan.breakfast),
                ("lunch", plan.lunch),
                ("dinner", plan.dinner),
            ]:
                # Evaluate complexity based on cooking instructions
                instructions = meal.get("cooking_instructions", "")
                complexity = self._calculate_cooking_complexity(instructions)

                # Breakfast should be simpler
                if meal_type == "breakfast":
                    complexity_score = 1.0 if complexity <= 2 else 0.7
                else:
                    complexity_score = 1.0 if complexity <= 5 else 0.8

                # Ingredient count should be reasonable (3-8 ingredients)
                ingredient_count = len(meal.get("ingredients", []))
                ingredient_score = 1.0 if 3 <= ingredient_count <= 8 else 0.7

                # Time estimation based on instructions
                estimated_time = self._estimate_cooking_time(instructions)
                time_score = self._evaluate_time_appropriateness(
                    meal_type, estimated_time
                )

                meal_score = (complexity_score + ingredient_score + time_score) / 3
                day_scores.append(meal_score)

                complexity_details.append(
                    {
                        "day": plan.day,
                        "meal_type": meal_type,
                        "complexity": complexity,
                        "estimated_time_minutes": estimated_time,
                        "ingredient_count": ingredient_count,
                    }
                )
                ingredient_count_details.append(ingredient_count)

            feasibility_scores.append(np.mean(day_scores))

        overall_score = (
            float(np.mean(feasibility_scores)) if feasibility_scores else 0.0
        )

        details = {
            "average_feasibility": overall_score,
            "daily_scores": feasibility_scores,
            "complexity_details": complexity_details,
            "avg_ingredient_count": np.mean(ingredient_count_details)
            if ingredient_count_details
            else 0,
            "avg_cooking_time": np.mean(
                [d["estimated_time_minutes"] for d in complexity_details]
            ),
        }

        return overall_score, details

    def _calculate_cooking_complexity(self, instructions: str) -> float:
        """Calculate cooking complexity based on instructions."""
        if not instructions:
            return 1.0

        # Base complexity from instruction length
        word_count = len(instructions.split())
        base_complexity = min(word_count / 20, 3.0)

        # Add complexity for specific cooking techniques
        complexity_keywords = {
            "marinate": 0.5,
            "slow cook": 1.0,
            "braise": 0.8,
            "roast": 0.6,
            "grill": 0.4,
            "saute": 0.3,
            "simmer": 0.4,
            "steam": 0.2,
            "deep fry": 0.8,
            "stir fry": 0.3,
            "bake": 0.5,
        }

        technique_complexity = 0
        instructions_lower = instructions.lower()
        for technique, complexity in complexity_keywords.items():
            if technique in instructions_lower:
                technique_complexity += complexity

        return min(base_complexity + technique_complexity, 5.0)

    def _estimate_cooking_time(self, instructions: str) -> int:
        """Estimate cooking time in minutes based on instructions."""
        if not instructions:
            return 15

        instructions_lower = instructions.lower()

        # Look for explicit time mentions
        import re

        time_patterns = [
            r"(\d+)\s*minutes?",
            r"(\d+)\s*mins?",
            r"(\d+)\s*hours?",
            r"(\d+)\s*hrs?",
        ]

        total_time = 0
        for pattern in time_patterns:
            matches = re.findall(pattern, instructions_lower)
            for match in matches:
                time_value = int(match)
                if "hour" in pattern or "hr" in pattern:
                    time_value *= 60
                total_time += time_value

        if total_time > 0:
            return total_time

        # Estimate based on cooking methods
        time_estimates = {
            "marinate": 30,
            "slow cook": 120,
            "braise": 90,
            "roast": 60,
            "grill": 20,
            "saute": 10,
            "simmer": 30,
            "steam": 15,
            "deep fry": 15,
            "stir fry": 10,
            "bake": 45,
            "boil": 15,
        }

        estimated_time = 15  # Base time
        for method, time in time_estimates.items():
            if method in instructions_lower:
                estimated_time = max(estimated_time, time)

        return estimated_time

    def _evaluate_time_appropriateness(
        self, meal_type: str, estimated_time: int
    ) -> float:
        """Evaluate if cooking time is appropriate for meal type."""
        if meal_type == "breakfast":
            # Breakfast should be quick (under 20 minutes ideal)
            if estimated_time <= 20:
                return 1.0
            elif estimated_time <= 30:
                return 0.8
            else:
                return 0.6
        elif meal_type == "lunch":
            # Lunch should be moderate (under 40 minutes ideal)
            if estimated_time <= 40:
                return 1.0
            elif estimated_time <= 60:
                return 0.9
            else:
                return 0.7
        else:  # dinner
            # Dinner can be longer (under 90 minutes ideal)
            if estimated_time <= 90:
                return 1.0
            elif estimated_time <= 120:
                return 0.9
            else:
                return 0.8


class NutritionalBalanceEvaluator(QualityEvaluator):
    """Evaluates nutritional balance across meals and days."""

    @property
    def name(self) -> str:
        return "nutritional_balance"

    @property
    def description(self) -> str:
        return "Evaluates consistency and balance of nutrition across meals and days"

    def evaluate(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate nutritional balance and consistency."""
        daily_calories = []
        daily_protein_ratios = []
        daily_fat_ratios = []
        daily_carb_ratios = []

        for plan in meal_plans:
            total_calories = plan.daily_nutrition.get("total_calories", 0)
            total_protein = plan.daily_nutrition.get("total_protein_g", 0)
            total_fat = plan.daily_nutrition.get("total_fat_g", 0)
            total_carbs = plan.daily_nutrition.get("total_carbs_g", 0)

            daily_calories.append(total_calories)

            # Calculate PFC ratios
            if total_calories > 0:
                protein_cal = total_protein * 4
                fat_cal = total_fat * 9
                carb_cal = total_carbs * 4
                total_macro_cal = protein_cal + fat_cal + carb_cal

                if total_macro_cal > 0:
                    daily_protein_ratios.append(protein_cal / total_macro_cal)
                    daily_fat_ratios.append(fat_cal / total_macro_cal)
                    daily_carb_ratios.append(carb_cal / total_macro_cal)

        # Calculate consistency scores
        calorie_consistency = self._calculate_consistency_score(daily_calories)
        protein_consistency = self._calculate_consistency_score(daily_protein_ratios)
        fat_consistency = self._calculate_consistency_score(daily_fat_ratios)
        carb_consistency = self._calculate_consistency_score(daily_carb_ratios)

        # Overall balance score
        overall_score = (
            calorie_consistency * 0.4
            + protein_consistency * 0.2
            + fat_consistency * 0.2
            + carb_consistency * 0.2
        )

        details = {
            "calorie_consistency": calorie_consistency,
            "protein_consistency": protein_consistency,
            "fat_consistency": fat_consistency,
            "carb_consistency": carb_consistency,
            "daily_calories": daily_calories,
            "avg_daily_calories": np.mean(daily_calories) if daily_calories else 0,
            "calorie_std": np.std(daily_calories) if daily_calories else 0,
            "daily_protein_ratios": daily_protein_ratios,
            "daily_fat_ratios": daily_fat_ratios,
            "daily_carb_ratios": daily_carb_ratios,
        }

        return overall_score, details

    def _calculate_consistency_score(self, values: list[float]) -> float:
        """Calculate consistency score based on standard deviation."""
        if len(values) <= 1:
            return 1.0

        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 1.0 if std_val == 0 else 0.0
        else:
            # Coefficient of variation
            cv = float(std_val / mean_val)

            # Score decreases as variation increases
            # CV of 0.1 (10%) gets score of 1.0, CV of 0.3 (30%) gets score of 0.0
            consistency_score = max(0.0, 1.0 - (cv / 0.3))

            return consistency_score
