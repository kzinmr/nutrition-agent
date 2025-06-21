import asyncio
import base64
import hashlib
import hmac
import json
import os
import random
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class NutritionInfo:
    calories: float
    protein: float  # in grams
    fat: float  # in grams
    carbs: float  # in grams

    @property
    def pfc_ratio(self) -> tuple[float, float, float]:
        total_calories = (self.protein * 4) + (self.fat * 9) + (self.carbs * 4)
        if total_calories == 0:
            return (0.0, 0.0, 0.0)

        protein_pct = (self.protein * 4) / total_calories * 100
        fat_pct = (self.fat * 9) / total_calories * 100
        carbs_pct = (self.carbs * 4) / total_calories * 100

        return (protein_pct, fat_pct, carbs_pct)


@dataclass
class Food:
    food_id: str
    food_name: str
    nutrition_per_100g: NutritionInfo
    brand_name: str | None = None


@dataclass
class Recipe:
    recipe_id: str
    recipe_name: str
    recipe_description: str
    recipe_nutrition: NutritionInfo
    ingredients: list[str]


class FatSecretAPIError(Exception):
    pass


class FatSecretClient:
    BASE_URL = "https://platform.fatsecret.com/rest/server.api"

    def __init__(
        self, consumer_key: str | None = None, consumer_secret: str | None = None
    ):
        self.consumer_key = consumer_key or os.getenv("FATSECRET_CONSUMER_KEY")
        self.consumer_secret = consumer_secret or os.getenv("FATSECRET_CONSUMER_SECRET")

        if not self.consumer_key or not self.consumer_secret:
            raise ValueError(
                "FatSecret API credentials not found. Please set FATSECRET_CONSUMER_KEY and FATSECRET_CONSUMER_SECRET"
            )

        self._cache: dict[str, Any] = {}
        self._last_request_time: float = 0
        self._min_request_interval = 0.1  # Rate limiting

    def _generate_oauth_signature(
        self, method: str, url: str, params: dict[str, str]
    ) -> str:
        """Generate OAuth 1.0a HMAC-SHA1 signature"""
        # Encode parameters
        encoded_params = {}
        for key, value in params.items():
            encoded_params[key] = urllib.parse.quote(str(value), safe="")

        # Sort and concatenate parameters
        sorted_params = sorted(encoded_params.items())
        param_string = "&".join([f"{k}={v}" for k, v in sorted_params])

        # Create signature base string
        signature_base = "&".join(
            [
                method.upper(),
                urllib.parse.quote(url, safe=""),
                urllib.parse.quote(param_string, safe=""),
            ]
        )

        # Create signing key
        signing_key = f"{self.consumer_secret}&"

        # Generate HMAC-SHA1 signature
        hashed = hmac.new(
            signing_key.encode("utf-8"), signature_base.encode("utf-8"), hashlib.sha1
        )

        # Base64 encode the signature
        signature = base64.b64encode(hashed.digest()).decode("utf-8")

        return signature

    async def _make_request(
        self, method: str, params: dict[str, str]
    ) -> dict[str, Any]:
        cache_key = f"{method}:{json.dumps(params, sort_keys=True)}"

        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[no-any-return]

        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last_request)

        # Prepare OAuth parameters
        oauth_params = {
            "oauth_consumer_key": self.consumer_key,
            "oauth_nonce": str(random.randint(10000000, 99999999)),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_version": "1.0",
            "format": "json",
            "method": method,
        }

        # Combine OAuth and API parameters
        all_params = {**oauth_params, **params}

        # Generate OAuth signature
        signature = self._generate_oauth_signature("GET", self.BASE_URL, all_params)

        # Add signature to parameters
        all_params["oauth_signature"] = signature

        try:
            # Make GET request with all parameters
            response = requests.get(self.BASE_URL, params=all_params)
            response.raise_for_status()

            self._last_request_time = time.time()
            data = response.json()

            # Check for API errors
            if "error" in data:
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise FatSecretAPIError(f"API Error: {error_msg}")

            self._cache[cache_key] = data
            return data  # type: ignore[no-any-return]

        except requests.exceptions.HTTPError as e:
            raise FatSecretAPIError(
                f"HTTP Error {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise FatSecretAPIError(f"Request failed: {str(e)}")

    async def search_food(self, query: str, max_results: int = 10) -> list[Food]:
        params = {"search_expression": query, "max_results": str(max_results)}

        data = await self._make_request("foods.search", params)

        foods = []
        if "foods" in data and "food" in data["foods"]:
            food_list = data["foods"]["food"]
            if not isinstance(food_list, list):
                food_list = [food_list]

            for food_data in food_list[:max_results]:
                try:
                    food_id = food_data["food_id"]
                    food_detail = await self.get_food(food_id)
                    if food_detail:
                        foods.append(food_detail)
                except Exception:
                    continue

        return foods

    async def get_food(self, food_id: str) -> Food | None:
        params = {"food_id": food_id}

        try:
            data = await self._make_request("food.get", params)

            if "food" not in data:
                return None

            food_data = data["food"]

            # Extract nutrition info from servings
            servings = food_data.get("servings", {}).get("serving", [])
            if not isinstance(servings, list):
                servings = [servings]

            # Find the 100g serving or calculate from available serving
            nutrition_info = None
            for serving in servings:
                if serving.get("serving_description") == "100 g":
                    nutrition_info = NutritionInfo(
                        calories=float(serving.get("calories", 0)),
                        protein=float(serving.get("protein", 0)),
                        fat=float(serving.get("fat", 0)),
                        carbs=float(serving.get("carbohydrate", 0)),
                    )
                    break

            # If no 100g serving, calculate from first available serving
            if not nutrition_info and servings:
                serving = servings[0]
                metric_serving_amount = float(serving.get("metric_serving_amount", 100))
                if metric_serving_amount > 0:
                    scale_factor = 100 / metric_serving_amount
                    nutrition_info = NutritionInfo(
                        calories=float(serving.get("calories", 0)) * scale_factor,
                        protein=float(serving.get("protein", 0)) * scale_factor,
                        fat=float(serving.get("fat", 0)) * scale_factor,
                        carbs=float(serving.get("carbohydrate", 0)) * scale_factor,
                    )

            if not nutrition_info:
                return None

            return Food(
                food_id=food_id,
                food_name=food_data.get("food_name", ""),
                brand_name=food_data.get("brand_name"),
                nutrition_per_100g=nutrition_info,
            )

        except Exception:
            return None

    async def search_recipes(self, query: str, max_results: int = 10) -> list[Recipe]:
        params = {"search_expression": query, "max_results": str(max_results)}

        data = await self._make_request("recipes.search", params)

        recipes = []
        if "recipes" in data and "recipe" in data["recipes"]:
            recipe_list = data["recipes"]["recipe"]
            if not isinstance(recipe_list, list):
                recipe_list = [recipe_list]

            for recipe_data in recipe_list[:max_results]:
                try:
                    recipe_id = recipe_data["recipe_id"]
                    recipe_detail = await self.get_recipe(recipe_id)
                    if recipe_detail:
                        recipes.append(recipe_detail)
                except Exception:
                    continue

        return recipes

    async def get_recipe(self, recipe_id: str) -> Recipe | None:
        params = {"recipe_id": recipe_id}

        try:
            data = await self._make_request("recipe.get", params)

            if "recipe" not in data:
                return None

            recipe_data = data["recipe"]

            # Extract nutrition info
            serving = recipe_data.get("serving_sizes", {}).get("serving", {})
            if isinstance(serving, list):
                serving = serving[0] if serving else {}

            nutrition_info = NutritionInfo(
                calories=float(serving.get("calories", 0)),
                protein=float(serving.get("protein", 0)),
                fat=float(serving.get("fat", 0)),
                carbs=float(serving.get("carbohydrate", 0)),
            )

            # Extract ingredients
            ingredients = []
            ingredients_data = recipe_data.get("ingredients", {}).get("ingredient", [])
            if not isinstance(ingredients_data, list):
                ingredients_data = [ingredients_data]

            for ingredient in ingredients_data:
                ingredients.append(ingredient.get("ingredient_description", ""))

            return Recipe(
                recipe_id=recipe_id,
                recipe_name=recipe_data.get("recipe_name", ""),
                recipe_description=recipe_data.get("recipe_description", ""),
                recipe_nutrition=nutrition_info,
                ingredients=ingredients,
            )

        except Exception:
            return None

    def clear_cache(self) -> None:
        self._cache.clear()


# Tool functions for agent use
async def search_food_nutrition(food_name: str) -> dict[str, Any]:
    """Search for food items and get their nutritional information."""
    client = FatSecretClient()
    foods = await client.search_food(food_name, max_results=5)

    results = []
    for food in foods:
        pfc = food.nutrition_per_100g.pfc_ratio
        results.append(
            {
                "name": food.food_name,
                "brand": food.brand_name,
                "nutrition_per_100g": {
                    "calories": food.nutrition_per_100g.calories,
                    "protein_g": food.nutrition_per_100g.protein,
                    "fat_g": food.nutrition_per_100g.fat,
                    "carbs_g": food.nutrition_per_100g.carbs,
                    "pfc_ratio": {
                        "protein_pct": round(pfc[0], 1),
                        "fat_pct": round(pfc[1], 1),
                        "carbs_pct": round(pfc[2], 1),
                    },
                },
            }
        )

    return {"foods": results}


async def search_recipes_by_ingredients(
    ingredients: list[str], dietary_restrictions: list[str] | None = None
) -> dict[str, Any]:
    """Search for recipes based on ingredients and dietary restrictions."""
    client = FatSecretClient()

    # Build search query
    query = " ".join(ingredients)
    if dietary_restrictions:
        query += " " + " ".join(dietary_restrictions)

    recipes = await client.search_recipes(query, max_results=5)

    results = []
    for recipe in recipes:
        pfc = recipe.recipe_nutrition.pfc_ratio
        results.append(
            {
                "name": recipe.recipe_name,
                "description": recipe.recipe_description,
                "ingredients": recipe.ingredients,
                "nutrition": {
                    "calories": recipe.recipe_nutrition.calories,
                    "protein_g": recipe.recipe_nutrition.protein,
                    "fat_g": recipe.recipe_nutrition.fat,
                    "carbs_g": recipe.recipe_nutrition.carbs,
                    "pfc_ratio": {
                        "protein_pct": round(pfc[0], 1),
                        "fat_pct": round(pfc[1], 1),
                        "carbs_pct": round(pfc[2], 1),
                    },
                },
            }
        )

    return {"recipes": results}
