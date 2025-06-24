from dataclasses import dataclass
from typing import Any

import numpy as np

# Nutritional constants: calories per gram for each macronutrient
CAL_PER_GRAM_P = 4  # Protein provides 4 kcal/g
CAL_PER_GRAM_F = 9  # Fat provides 9 kcal/g
CAL_PER_GRAM_C = 4  # Carbohydrates provide 4 kcal/g


@dataclass
class NutritionTarget:
    daily_calories: float
    pfc_ratio: tuple[float, float, float]  # Protein%, Fat%, Carbs%

    @property
    def daily_protein_g(self) -> float:
        return (self.daily_calories * self.pfc_ratio[0] / 100) / CAL_PER_GRAM_P

    @property
    def daily_fat_g(self) -> float:
        return (self.daily_calories * self.pfc_ratio[1] / 100) / CAL_PER_GRAM_F

    @property
    def daily_carbs_g(self) -> float:
        return (self.daily_calories * self.pfc_ratio[2] / 100) / CAL_PER_GRAM_C


@dataclass
class MealNutrition:
    meal_name: str
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float

    @property
    def pfc_ratio(self) -> tuple[float, float, float]:
        total_calories = (
            (self.protein_g * CAL_PER_GRAM_P)
            + (self.fat_g * CAL_PER_GRAM_F)
            + (self.carbs_g * CAL_PER_GRAM_C)
        )
        if total_calories == 0:
            return (0.0, 0.0, 0.0)

        protein_pct = (self.protein_g * CAL_PER_GRAM_P) / total_calories * 100
        fat_pct = (self.fat_g * CAL_PER_GRAM_F) / total_calories * 100
        carbs_pct = (self.carbs_g * CAL_PER_GRAM_C) / total_calories * 100

        return (protein_pct, fat_pct, carbs_pct)


class NutritionCalculator:
    @staticmethod
    def calculate_meal_nutrition(ingredients: list[dict[str, float]]) -> MealNutrition:
        """
        Calculate total nutrition for a meal from ingredients.

        Args:
            ingredients: List of dicts with keys:
                - name: str
                - amount_g: float
                - calories_per_100g: float
                - protein_per_100g: float
                - fat_per_100g: float
                - carbs_per_100g: float
        """
        total_calories: float = 0
        total_protein: float = 0
        total_fat: float = 0
        total_carbs: float = 0

        for ingredient in ingredients:
            amount_factor = ingredient["amount_g"] / 100
            total_calories += ingredient["calories_per_100g"] * amount_factor
            total_protein += ingredient["protein_per_100g"] * amount_factor
            total_fat += ingredient["fat_per_100g"] * amount_factor
            total_carbs += ingredient["carbs_per_100g"] * amount_factor

        return MealNutrition(
            meal_name="Combined meal",
            calories=total_calories,
            protein_g=total_protein,
            fat_g=total_fat,
            carbs_g=total_carbs,
        )

    @staticmethod
    def calculate_daily_nutrition(meals: list[MealNutrition]) -> MealNutrition:
        """Calculate total nutrition for a day from multiple meals."""
        total_calories = sum(meal.calories for meal in meals)
        total_protein = sum(meal.protein_g for meal in meals)
        total_fat = sum(meal.fat_g for meal in meals)
        total_carbs = sum(meal.carbs_g for meal in meals)

        return MealNutrition(
            meal_name="Daily total",
            calories=total_calories,
            protein_g=total_protein,
            fat_g=total_fat,
            carbs_g=total_carbs,
        )

    @staticmethod
    def calculate_nutrition_error(
        actual: MealNutrition, target: NutritionTarget
    ) -> dict[str, float]:
        """
        Calculate percentage error between actual nutrition and target.

        Returns:
            Dict with keys: calories_error, protein_error, fat_error, carbs_error, pfc_error
            All values are percentages (0-100)
        """
        calories_error = (
            abs(actual.calories - target.daily_calories) / target.daily_calories * 100
        )
        protein_error = (
            abs(actual.protein_g - target.daily_protein_g)
            / target.daily_protein_g
            * 100
        )
        fat_error = abs(actual.fat_g - target.daily_fat_g) / target.daily_fat_g * 100
        carbs_error = (
            abs(actual.carbs_g - target.daily_carbs_g) / target.daily_carbs_g * 100
        )

        # PFC ratio error (average of individual macro errors)
        actual_pfc = actual.pfc_ratio
        target_pfc = target.pfc_ratio

        pfc_errors = [
            abs(actual_pfc[0] - target_pfc[0]),
            abs(actual_pfc[1] - target_pfc[1]),
            abs(actual_pfc[2] - target_pfc[2]),
        ]
        pfc_error = float(np.mean(pfc_errors))

        return {
            "calories_error": calories_error,
            "protein_error": protein_error,
            "fat_error": fat_error,
            "carbs_error": carbs_error,
            "pfc_error": pfc_error,
        }

    @staticmethod
    def check_nutrition_constraints(
        actual: MealNutrition, target: NutritionTarget, tolerance_pct: float = 10.0
    ) -> tuple[bool, list[str]]:
        """
        Check if nutrition meets target within tolerance.

        Returns:
            Tuple of (meets_constraints: bool, violations: List[str])
        """
        errors = NutritionCalculator.calculate_nutrition_error(actual, target)
        violations = []

        if errors["calories_error"] > tolerance_pct:
            violations.append(f"Calories off by {errors['calories_error']:.1f}%")

        if errors["protein_error"] > tolerance_pct:
            violations.append(f"Protein off by {errors['protein_error']:.1f}%")

        if errors["fat_error"] > tolerance_pct:
            violations.append(f"Fat off by {errors['fat_error']:.1f}%")

        if errors["carbs_error"] > tolerance_pct:
            violations.append(f"Carbs off by {errors['carbs_error']:.1f}%")

        meets_constraints = len(violations) == 0
        return meets_constraints, violations

    @staticmethod
    def suggest_portion_adjustments(
        meal: MealNutrition,
        target: NutritionTarget,
        meal_fraction: float = 0.33,  # Fraction of daily target for this meal
        tolerance_ratio: float = 0.1,
        # Tolerance ratio for meeting individual macro targets
    ) -> dict[str, float]:
        """
        Suggest how to adjust portion sizes to meet targets.

        Returns:
            Dict with adjustment factors for each macro
        """
        target_meal_calories = target.daily_calories * meal_fraction
        target_meal_protein = target.daily_protein_g * meal_fraction
        target_meal_fat = target.daily_fat_g * meal_fraction
        target_meal_carbs = target.daily_carbs_g * meal_fraction

        # Calculate optimal scaling factor based on calories
        scale_factor = (
            target_meal_calories / meal.calories if meal.calories > 0 else 1.0
        )

        # Check if scaling would meet all targets
        scaled_protein = meal.protein_g * scale_factor
        scaled_fat = meal.fat_g * scale_factor
        scaled_carbs = meal.carbs_g * scale_factor

        return {
            "overall_scale": scale_factor,
            "scaled_calories": meal.calories * scale_factor,
            "scaled_protein_g": scaled_protein,
            "scaled_fat_g": scaled_fat,
            "scaled_carbs_g": scaled_carbs,
            "meets_protein": abs(scaled_protein - target_meal_protein)
            / target_meal_protein
            < tolerance_ratio,
            "meets_fat": abs(scaled_fat - target_meal_fat) / target_meal_fat
            < tolerance_ratio,
            "meets_carbs": abs(scaled_carbs - target_meal_carbs) / target_meal_carbs
            < tolerance_ratio,
        }


# Tool functions for agent use
def calculate_pfc_balance(
    meals: list[dict[str, Any]],
    target_calories: float,
    target_pfc: tuple[float, float, float],
) -> dict[str, Any]:
    """Calculate PFC balance for a set of meals and compare to targets."""

    # Convert meal data to MealNutrition objects
    meal_nutritions = []
    for meal in meals:
        meal_nutritions.append(
            MealNutrition(
                meal_name=meal["name"],
                calories=meal["calories"],
                protein_g=meal["protein_g"],
                fat_g=meal["fat_g"],
                carbs_g=meal["carbs_g"],
            )
        )

    # Calculate daily totals
    calculator = NutritionCalculator()
    daily_nutrition = calculator.calculate_daily_nutrition(meal_nutritions)

    # Create target
    target = NutritionTarget(daily_calories=target_calories, pfc_ratio=target_pfc)

    # Calculate errors
    errors = calculator.calculate_nutrition_error(daily_nutrition, target)
    meets_constraints, violations = calculator.check_nutrition_constraints(
        daily_nutrition, target
    )

    return {
        "daily_totals": {
            "calories": daily_nutrition.calories,
            "protein_g": daily_nutrition.protein_g,
            "fat_g": daily_nutrition.fat_g,
            "carbs_g": daily_nutrition.carbs_g,
            "pfc_ratio": daily_nutrition.pfc_ratio,
        },
        "target": {
            "calories": target.daily_calories,
            "protein_g": target.daily_protein_g,
            "fat_g": target.daily_fat_g,
            "carbs_g": target.daily_carbs_g,
            "pfc_ratio": target.pfc_ratio,
        },
        "errors": errors,
        "meets_constraints": meets_constraints,
        "violations": violations,
    }
