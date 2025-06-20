import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from agents.base_agent import BaseAgent, AgentConfig, ModelProvider
from tools.fatsecret_tool import search_food_nutrition, search_recipes_by_ingredients
from tools.nutrition_calculator import calculate_pfc_balance, NutritionTarget, MealNutrition
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class Inventory:
    items: List[Dict[str, Any]]  # {"name": str, "amount_g": float, "unit": str}
    

@dataclass
class DietaryConstraints:
    daily_calories: float
    pfc_ratio: Tuple[float, float, float]  # Protein%, Fat%, Carbs%
    allergens: List[str] = None
    dietary_restrictions: List[str] = None  # vegetarian, vegan, low-carb, etc.
    

@dataclass
class MealPlan:
    day: int
    breakfast: Dict[str, Any]
    lunch: Dict[str, Any]
    dinner: Dict[str, Any]
    daily_nutrition: Dict[str, Any]
    missing_ingredients: List[str]


class NutritionPlannerAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._register_nutrition_tools()
        
    def _register_nutrition_tools(self):
        """Register nutrition-specific tools."""
        
        # Food nutrition search
        self.register_tool(
            name="search_food_nutrition",
            func=search_food_nutrition,
            description="Search for nutritional information of food items",
            parameters={
                "type": "object",
                "properties": {
                    "food_name": {
                        "type": "string",
                        "description": "Name of the food item to search"
                    }
                },
                "required": ["food_name"]
            }
        )
        
        # Recipe search
        self.register_tool(
            name="search_recipes_by_ingredients",
            func=search_recipes_by_ingredients,
            description="Search for recipes based on available ingredients",
            parameters={
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of available ingredients"
                    },
                    "dietary_restrictions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dietary restrictions (optional)"
                    }
                },
                "required": ["ingredients"]
            }
        )
        
        # PFC balance calculator
        self.register_tool(
            name="calculate_pfc_balance",
            func=calculate_pfc_balance,
            description="Calculate protein-fat-carbohydrate balance for meals",
            parameters={
                "type": "object",
                "properties": {
                    "meals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "calories": {"type": "number"},
                                "protein_g": {"type": "number"},
                                "fat_g": {"type": "number"},
                                "carbs_g": {"type": "number"}
                            },
                            "required": ["name", "calories", "protein_g", "fat_g", "carbs_g"]
                        },
                        "description": "List of meals with nutrition info"
                    },
                    "target_calories": {
                        "type": "number",
                        "description": "Target daily calories"
                    },
                    "target_pfc": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Target PFC ratio as [protein%, fat%, carbs%]"
                    }
                },
                "required": ["meals", "target_calories", "target_pfc"]
            }
        )
    
    async def generate_meal_plan(
        self,
        inventory: Inventory,
        constraints: DietaryConstraints,
        days: int = 3
    ) -> List[MealPlan]:
        """Generate a multi-day meal plan based on inventory and constraints."""
        
        # Format the user request
        inventory_text = "\n".join([
            f"- {item['name']}: {item['amount_g']}g"
            for item in inventory.items
        ])
        
        allergen_text = ""
        if constraints.allergens:
            allergen_text = f"\nAllergens to avoid: {', '.join(constraints.allergens)}"
        
        restriction_text = ""
        if constraints.dietary_restrictions:
            restriction_text = f"\nDietary restrictions: {', '.join(constraints.dietary_restrictions)}"
        
        user_request = f"""
        I need a {days}-day meal plan with the following requirements:
        
        Available ingredients:
        {inventory_text}
        
        Nutritional targets:
        - Daily calories: {constraints.daily_calories} kcal
        - PFC ratio: {constraints.pfc_ratio[0]}% protein, {constraints.pfc_ratio[1]}% fat, {constraints.pfc_ratio[2]}% carbohydrates
        {allergen_text}
        {restriction_text}
        
        Please create a complete meal plan that:
        1. Uses the available ingredients efficiently
        2. Meets the nutritional targets within ±10%
        3. Provides variety across days
        4. Lists any missing ingredients needed
        5. Includes portion sizes for each ingredient
        
        For each day, provide breakfast, lunch, and dinner with detailed nutritional information.
        """
        
        console.print(Panel(user_request, title="📋 [bold blue]Meal Planning Request[/bold blue]"))
        
        # Run the agent
        response = await self.run(user_request)
        
        # Parse the response to extract meal plans
        meal_plans = self._parse_meal_plan_response(response, days)
        
        return meal_plans
    
    def _parse_meal_plan_response(self, response: str, days: int) -> List[MealPlan]:
        """Parse the agent's response into structured meal plans."""
        # This is a simplified parser - in production, you'd want more robust parsing
        # or use structured output from the LLM
        
        meal_plans = []
        
        # For now, return a mock parsed result
        # In a real implementation, you would parse the actual LLM response
        for day in range(1, days + 1):
            meal_plan = MealPlan(
                day=day,
                breakfast={
                    "name": f"Day {day} Breakfast",
                    "ingredients": [],
                    "calories": 400,
                    "protein_g": 20,
                    "fat_g": 15,
                    "carbs_g": 45
                },
                lunch={
                    "name": f"Day {day} Lunch",
                    "ingredients": [],
                    "calories": 600,
                    "protein_g": 30,
                    "fat_g": 20,
                    "carbs_g": 65
                },
                dinner={
                    "name": f"Day {day} Dinner",
                    "ingredients": [],
                    "calories": 700,
                    "protein_g": 35,
                    "fat_g": 25,
                    "carbs_g": 75
                },
                daily_nutrition={
                    "total_calories": 1700,
                    "total_protein_g": 85,
                    "total_fat_g": 60,
                    "total_carbs_g": 185,
                    "pfc_ratio": [30, 25, 45]
                },
                missing_ingredients=["carrots", "olive oil", "whole wheat bread"]
            )
            meal_plans.append(meal_plan)
        
        return meal_plans
    
    def display_meal_plans(self, meal_plans: List[MealPlan]):
        """Display meal plans in a formatted table."""
        for plan in meal_plans:
            # Day header
            console.print(f"\n[bold cyan]Day {plan.day} Meal Plan[/bold cyan]")
            
            # Meals table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Meal", style="cyan", width=12)
            table.add_column("Calories", justify="right")
            table.add_column("Protein (g)", justify="right")
            table.add_column("Fat (g)", justify="right")
            table.add_column("Carbs (g)", justify="right")
            
            # Add meals
            for meal_type, meal_data in [
                ("Breakfast", plan.breakfast),
                ("Lunch", plan.lunch),
                ("Dinner", plan.dinner)
            ]:
                table.add_row(
                    meal_type,
                    str(meal_data["calories"]),
                    str(meal_data["protein_g"]),
                    str(meal_data["fat_g"]),
                    str(meal_data["carbs_g"])
                )
            
            # Daily total
            table.add_row(
                "[bold]Daily Total[/bold]",
                f"[bold]{plan.daily_nutrition['total_calories']}[/bold]",
                f"[bold]{plan.daily_nutrition['total_protein_g']}[/bold]",
                f"[bold]{plan.daily_nutrition['total_fat_g']}[/bold]",
                f"[bold]{plan.daily_nutrition['total_carbs_g']}[/bold]",
                style="green"
            )
            
            console.print(table)
            
            # PFC ratio
            pfc = plan.daily_nutrition['pfc_ratio']
            console.print(f"PFC Ratio: [yellow]{pfc[0]:.1f}% / {pfc[1]:.1f}% / {pfc[2]:.1f}%[/yellow]")
            
            # Missing ingredients
            if plan.missing_ingredients:
                console.print(f"Missing ingredients: [red]{', '.join(plan.missing_ingredients)}[/red]")
    
    def generate_shopping_list(self, meal_plans: List[MealPlan]) -> Dict[str, List[str]]:
        """Generate a consolidated shopping list from meal plans."""
        shopping_list = {
            "proteins": set(),
            "vegetables": set(),
            "grains": set(),
            "dairy": set(),
            "others": set()
        }
        
        # Collect all missing ingredients
        for plan in meal_plans:
            for ingredient in plan.missing_ingredients:
                # Simple categorization - in production, use a proper food database
                if any(word in ingredient.lower() for word in ["chicken", "beef", "fish", "tofu"]):
                    shopping_list["proteins"].add(ingredient)
                elif any(word in ingredient.lower() for word in ["carrot", "broccoli", "lettuce", "tomato"]):
                    shopping_list["vegetables"].add(ingredient)
                elif any(word in ingredient.lower() for word in ["rice", "bread", "pasta", "oats"]):
                    shopping_list["grains"].add(ingredient)
                elif any(word in ingredient.lower() for word in ["milk", "cheese", "yogurt"]):
                    shopping_list["dairy"].add(ingredient)
                else:
                    shopping_list["others"].add(ingredient)
        
        # Convert sets to lists
        return {k: list(v) for k, v in shopping_list.items() if v}
    
    def display_shopping_list(self, shopping_list: Dict[str, List[str]]):
        """Display shopping list in a formatted way."""
        console.print("\n[bold green]Shopping List[/bold green]")
        
        for category, items in shopping_list.items():
            if items:
                console.print(f"\n[yellow]{category.title()}:[/yellow]")
                for item in items:
                    console.print(f"  • {item}")