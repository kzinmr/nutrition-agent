"""
Base classes for nutrition evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Any

from agents.nutrition_planner import DietaryConstraints, Inventory, MealPlan


class BaseEvaluator(ABC):
    """Base class for all nutrition evaluators."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def evaluate(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate meal plans and return score and details.

        Args:
            meal_plans: List of meal plans to evaluate
            constraints: Dietary constraints
            inventory: Available inventory

        Returns:
            Tuple of (score, details) where score is 0.0-1.0
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this evaluator."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this evaluator measures."""
        pass


class MandatoryEvaluator(BaseEvaluator):
    """Base class for mandatory evaluators that can cause immediate failure."""

    @abstractmethod
    def is_critical_failure(
        self,
        meal_plans: list[MealPlan],
        constraints: DietaryConstraints,
        inventory: Inventory,
    ) -> bool:
        """
        Check if this evaluation represents a critical failure.

        Returns:
            True if this is a critical failure that should result in score = 0
        """
        pass


class QualityEvaluator(BaseEvaluator):
    """Base class for quality evaluators that assess meal plan quality."""

    pass
