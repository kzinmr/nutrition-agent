import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.table import Table

from agents.base_agent import AgentConfig, BaseAgent
from tools.fatsecret_tool import search_food_nutrition, search_recipes_by_ingredients
from tools.nutrition_calculator import calculate_pfc_balance

console = Console()


# Pydantic models for structured output
class MealStructured(BaseModel):
    """Structured representation of a meal for LLM output."""

    name: str
    ingredients: list[str]
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float
    cooking_instructions: str = ""


class DailyNutritionStructured(BaseModel):
    """Structured representation of daily nutrition totals."""

    total_calories: float
    total_protein_g: float
    total_fat_g: float
    total_carbs_g: float
    pfc_ratio: list[float] = Field(
        description="PFC ratio as [protein%, fat%, carbs%]", min_length=3, max_length=3
    )

    @field_validator("pfc_ratio")
    @classmethod
    def validate_pfc_ratio(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("PFC ratio must have exactly 3 values")
        if abs(sum(v) - 100.0) > 1.0:  # Allow 1% tolerance
            raise ValueError("PFC ratio percentages should sum to approximately 100%")
        return v


class MealPlanStructured(BaseModel):
    """Structured representation of a single day's meal plan for LLM output."""

    day: int
    breakfast: MealStructured
    lunch: MealStructured
    dinner: MealStructured
    daily_nutrition: DailyNutritionStructured
    missing_ingredients: list[str]
    notes: str = ""


class MealPlansResponse(BaseModel):
    """Structured response containing multiple days of meal plans."""

    meal_plans: list[MealPlanStructured]
    total_shopping_list: list[str]
    general_notes: str = ""


@dataclass
class Inventory:
    items: list[dict[str, Any]]  # {"name": str, "amount_g": float, "unit": str}


@dataclass
class DietaryConstraints:
    daily_calories: float
    pfc_ratio: tuple[float, float, float]  # Protein%, Fat%, Carbs%
    allergens: list[str] = field(default_factory=list)
    dietary_restrictions: list[str] = field(
        default_factory=list
    )  # vegetarian, vegan, low-carb, etc.


@dataclass
class MealPlan:
    day: int
    breakfast: dict[str, Any]
    lunch: dict[str, Any]
    dinner: dict[str, Any]
    daily_nutrition: dict[str, Any]
    missing_ingredients: list[str]


class NutritionPlannerAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._register_nutrition_tools()

    def _register_nutrition_tools(self) -> None:
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
                        "description": "Name of the food item to search",
                    }
                },
                "required": ["food_name"],
            },
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
                        "description": "List of available ingredients",
                    },
                    "dietary_restrictions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dietary restrictions (optional)",
                    },
                },
                "required": ["ingredients"],
            },
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
                                "carbs_g": {"type": "number"},
                            },
                            "required": [
                                "name",
                                "calories",
                                "protein_g",
                                "fat_g",
                                "carbs_g",
                            ],
                        },
                        "description": "List of meals with nutrition info",
                    },
                    "target_calories": {
                        "type": "number",
                        "description": "Target daily calories",
                    },
                    "target_pfc": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Target PFC ratio as [protein%, fat%, carbs%]",
                    },
                },
                "required": ["meals", "target_calories", "target_pfc"],
            },
        )

    async def generate_meal_plan(
        self,
        inventory: Inventory,
        constraints: DietaryConstraints,
        days: int = 3,
    ) -> list[MealPlan]:
        """Generate a multi-day meal plan based on inventory and constraints.
        This method uses the base agent's run method to handle tool calls automatically.
        """

        # Prepare inventory and constraints information
        inventory_text = "\n".join(
            [f"- {item['name']}: {item['amount_g']}g" for item in inventory.items]
        )

        # Create comprehensive prompt that encourages tool usage
        prompt = f"""
        Create a detailed {days}-day meal plan using the available tools and information provided.
        
        AVAILABLE INGREDIENTS:
        {inventory_text}
        
        NUTRITIONAL TARGETS:
        - Daily calories: {constraints.daily_calories} kcal
        - PFC ratio: {constraints.pfc_ratio[0]}% protein, {constraints.pfc_ratio[1]}% fat, {constraints.pfc_ratio[2]}% carbohydrates
        - Allergens to avoid: {", ".join(constraints.allergens) if constraints.allergens else "None"}
        - Dietary restrictions: {", ".join(constraints.dietary_restrictions) if constraints.dietary_restrictions else "None"}
        
        INSTRUCTIONS:
        1. First, use the search_food_nutrition tool to get accurate nutritional information for key available ingredients
        2. Use the search_recipes_by_ingredients tool to find recipes that can be made with available ingredients
        3. Use the calculate_pfc_balance tool to verify your meal plan meets the nutritional targets
        4. Create realistic meals with accurate nutrition calculations based on the tool results
        5. Provide detailed cooking instructions for each meal
        6. List any missing ingredients needed for the meal plan
        
        After gathering all necessary information and creating the meal plan, provide the final result in the following JSON format:
        {{
            "meal_plans": [
                {{
                    "day": 1,
                    "breakfast": {{
                        "name": "meal name",
                        "ingredients": ["ingredient1", "ingredient2"],
                        "calories": 300,
                        "protein_g": 20,
                        "fat_g": 10,
                        "carbs_g": 30,
                        "cooking_instructions": "detailed instructions"
                    }},
                    "lunch": {{ ... similar structure ... }},
                    "dinner": {{ ... similar structure ... }},
                    "daily_nutrition": {{
                        "total_calories": 2000,
                        "total_protein_g": 100,
                        "total_fat_g": 60,
                        "total_carbs_g": 250,
                        "pfc_ratio": [20.0, 27.0, 53.0]
                    }},
                    "missing_ingredients": ["ingredient1", "ingredient2"],
                    "notes": "any additional notes"
                }}
            ],
            "total_shopping_list": ["all missing ingredients"],
            "general_notes": "general notes about the meal plan"
        }}
        """

        # Use the base agent's run method
        console.print(
            "[yellow]Using base agent to gather nutrition info and create meal plan...[/yellow]"
        )

        # Run the agent with the prompt
        response = await self.run(prompt)

        # Parse the JSON response
        try:
            # Extract JSON from the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                meal_plans_data = json.loads(json_str)

                # Convert to MealPlansResponse structure
                structured_response = MealPlansResponse(**meal_plans_data)
                return self._convert_structured_to_meal_plans(structured_response)
            else:
                # If no JSON found, try using structured output
                console.print(
                    "[yellow]No valid JSON found in response, trying structured output...[/yellow]"
                )

                # Get the message history from the last run
                assert isinstance(self.client, AsyncOpenAI)

                # Use structured output with the existing message history
                structured_completion = await self.client.beta.chat.completions.parse(
                    model=self.config.model_name,
                    messages=self.messages,  # type: ignore[arg-type]
                    response_format=MealPlansResponse,
                    temperature=0,
                )

                parsed_response = structured_completion.choices[0].message.parsed
                if parsed_response:
                    return self._convert_structured_to_meal_plans(parsed_response)
                else:
                    raise ValueError(
                        "Failed to get structured response from OpenAI API"
                    )

        except json.JSONDecodeError as e:
            console.print(f"[red]Failed to parse JSON response: {e}[/red]")
            console.print(f"[red]Response was: {response}[/red]")
            raise ValueError(f"Failed to parse meal plan response: {e}")

    def _convert_structured_to_meal_plans(
        self, structured_response: MealPlansResponse
    ) -> list[MealPlan]:
        """Convert structured Pydantic response to MealPlan dataclasses."""
        meal_plans = []

        for plan_data in structured_response.meal_plans:
            # Convert Pydantic models to dictionaries for MealPlan dataclass
            breakfast_dict = {
                "name": plan_data.breakfast.name,
                "ingredients": plan_data.breakfast.ingredients,
                "calories": plan_data.breakfast.calories,
                "protein_g": plan_data.breakfast.protein_g,
                "fat_g": plan_data.breakfast.fat_g,
                "carbs_g": plan_data.breakfast.carbs_g,
                "cooking_instructions": plan_data.breakfast.cooking_instructions,
            }

            lunch_dict = {
                "name": plan_data.lunch.name,
                "ingredients": plan_data.lunch.ingredients,
                "calories": plan_data.lunch.calories,
                "protein_g": plan_data.lunch.protein_g,
                "fat_g": plan_data.lunch.fat_g,
                "carbs_g": plan_data.lunch.carbs_g,
                "cooking_instructions": plan_data.lunch.cooking_instructions,
            }

            dinner_dict = {
                "name": plan_data.dinner.name,
                "ingredients": plan_data.dinner.ingredients,
                "calories": plan_data.dinner.calories,
                "protein_g": plan_data.dinner.protein_g,
                "fat_g": plan_data.dinner.fat_g,
                "carbs_g": plan_data.dinner.carbs_g,
                "cooking_instructions": plan_data.dinner.cooking_instructions,
            }

            daily_nutrition_dict = {
                "total_calories": plan_data.daily_nutrition.total_calories,
                "total_protein_g": plan_data.daily_nutrition.total_protein_g,
                "total_fat_g": plan_data.daily_nutrition.total_fat_g,
                "total_carbs_g": plan_data.daily_nutrition.total_carbs_g,
                "pfc_ratio": tuple(
                    plan_data.daily_nutrition.pfc_ratio
                ),  # Convert list to tuple
            }

            meal_plan = MealPlan(
                day=plan_data.day,
                breakfast=breakfast_dict,
                lunch=lunch_dict,
                dinner=dinner_dict,
                daily_nutrition=daily_nutrition_dict,
                missing_ingredients=plan_data.missing_ingredients,
            )

            meal_plans.append(meal_plan)

        return meal_plans

    def display_meal_plans(self, meal_plans: list[MealPlan]) -> None:
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
                ("Dinner", plan.dinner),
            ]:
                table.add_row(
                    meal_type,
                    str(meal_data["calories"]),
                    str(meal_data["protein_g"]),
                    str(meal_data["fat_g"]),
                    str(meal_data["carbs_g"]),
                )

            # Daily total
            table.add_row(
                "[bold]Daily Total[/bold]",
                f"[bold]{plan.daily_nutrition['total_calories']}[/bold]",
                f"[bold]{plan.daily_nutrition['total_protein_g']}[/bold]",
                f"[bold]{plan.daily_nutrition['total_fat_g']}[/bold]",
                f"[bold]{plan.daily_nutrition['total_carbs_g']}[/bold]",
                style="green",
            )

            console.print(table)

            # PFC ratio
            pfc = plan.daily_nutrition["pfc_ratio"]
            console.print(
                f"PFC Ratio: [yellow]{pfc[0]:.1f}% / {pfc[1]:.1f}% / {pfc[2]:.1f}%[/yellow]"
            )

            # Missing ingredients
            if plan.missing_ingredients:
                console.print(
                    f"Missing ingredients: [red]{', '.join(plan.missing_ingredients)}[/red]"
                )

    def generate_shopping_list(
        self, meal_plans: list[MealPlan]
    ) -> dict[str, list[str]]:
        """Generate a consolidated shopping list from meal plans."""
        shopping_list: dict[str, set[str]] = {
            "proteins": set(),
            "vegetables": set(),
            "grains": set(),
            "dairy": set(),
            "others": set(),
        }

        # Collect all missing ingredients
        for plan in meal_plans:
            for ingredient in plan.missing_ingredients:
                # FIXME: Simple categorization - in production, use a proper food database
                if any(
                    word in ingredient.lower()
                    for word in ["chicken", "beef", "fish", "tofu"]
                ):
                    shopping_list["proteins"].add(ingredient)
                elif any(
                    word in ingredient.lower()
                    for word in ["carrot", "broccoli", "lettuce", "tomato"]
                ):
                    shopping_list["vegetables"].add(ingredient)
                elif any(
                    word in ingredient.lower()
                    for word in ["rice", "bread", "pasta", "oats"]
                ):
                    shopping_list["grains"].add(ingredient)
                elif any(
                    word in ingredient.lower() for word in ["milk", "cheese", "yogurt"]
                ):
                    shopping_list["dairy"].add(ingredient)
                else:
                    shopping_list["others"].add(ingredient)

        # Convert sets to lists
        return {k: list(v) for k, v in shopping_list.items() if v}

    def display_shopping_list(self, shopping_list: dict[str, list[str]]) -> None:
        """Display shopping list in a formatted way."""
        console.print("\n[bold green]Shopping List[/bold green]")

        for category, items in shopping_list.items():
            if items:
                console.print(f"\n[yellow]{category.title()}:[/yellow]")
                for item in items:
                    console.print(f"  • {item}")
