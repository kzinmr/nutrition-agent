#!/usr/bin/env python3
"""
FatSecret API Client のテストスクリプト
実際のAPIエンドポイントに接続して動作を確認します。
"""

import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tools.fatsecret_tool import (
    FatSecretClient,
    search_food_nutrition,
    search_recipes_by_ingredients,
)

load_dotenv()

console = Console()


async def test_food_search():
    """食品検索機能のテスト"""
    console.print(Panel("[bold blue]食品検索テスト[/bold blue]"))

    client = FatSecretClient()

    # テスト対象の食品
    test_foods = ["chicken breast", "rice", "apple", "broccoli", "salmon"]

    for food_name in test_foods:
        console.print(f"\n[yellow]Searching for: {food_name}[/yellow]")

        try:
            foods = await client.search_food(food_name, max_results=3)

            if foods:
                table = Table(title=f"Results for '{food_name}'")
                table.add_column("Food Name", style="cyan")
                table.add_column("Brand", style="green")
                table.add_column("Calories/100g", justify="right")
                table.add_column("Protein (g)", justify="right")
                table.add_column("Fat (g)", justify="right")
                table.add_column("Carbs (g)", justify="right")

                for food in foods:
                    table.add_row(
                        food.food_name,
                        food.brand_name or "N/A",
                        f"{food.nutrition_per_100g.calories:.1f}",
                        f"{food.nutrition_per_100g.protein:.1f}",
                        f"{food.nutrition_per_100g.fat:.1f}",
                        f"{food.nutrition_per_100g.carbs:.1f}",
                    )

                console.print(table)

                # PFC比率も表示
                if foods:
                    pfc = foods[0].nutrition_per_100g.pfc_ratio
                    console.print(
                        f"PFC Ratio for {foods[0].food_name}: "
                        f"P:{pfc[0]:.1f}% F:{pfc[1]:.1f}% C:{pfc[2]:.1f}%"
                    )
            else:
                console.print(f"[red]No results found for '{food_name}'[/red]")

        except Exception as e:
            console.print(f"[red]Error searching for '{food_name}': {e}[/red]")


async def test_recipe_search():
    """レシピ検索機能のテスト"""
    console.print(Panel("[bold blue]レシピ検索テスト[/bold blue]"))

    client = FatSecretClient()

    # テスト用の材料リスト
    test_ingredients = ["chicken", "rice", "vegetables"]

    console.print(
        f"[yellow]Searching recipes with: {', '.join(test_ingredients)}[/yellow]"
    )

    try:
        recipes = await client.search_recipes(" ".join(test_ingredients), max_results=3)

        if recipes:
            for i, recipe in enumerate(recipes, 1):
                console.print(f"\n[green]Recipe {i}: {recipe.recipe_name}[/green]")
                console.print(f"Description: {recipe.recipe_description[:100]}...")
                console.print(f"Calories: {recipe.recipe_nutrition.calories:.0f}")
                console.print(f"Protein: {recipe.recipe_nutrition.protein:.1f}g")
                console.print(f"Fat: {recipe.recipe_nutrition.fat:.1f}g")
                console.print(f"Carbs: {recipe.recipe_nutrition.carbs:.1f}g")

                pfc = recipe.recipe_nutrition.pfc_ratio
                console.print(
                    f"PFC Ratio: P:{pfc[0]:.1f}% F:{pfc[1]:.1f}% C:{pfc[2]:.1f}%"
                )

                console.print(f"Ingredients ({len(recipe.ingredients)}):")
                for j, ingredient in enumerate(recipe.ingredients[:5], 1):
                    console.print(f"  {j}. {ingredient}")
                if len(recipe.ingredients) > 5:
                    console.print(f"  ... and {len(recipe.ingredients) - 5} more")
        else:
            console.print("[red]No recipes found[/red]")

    except Exception as e:
        console.print(f"[red]Error searching recipes: {e}[/red]")


async def test_tool_functions():
    """ツール関数のテスト"""
    console.print(Panel("[bold blue]ツール関数テスト[/bold blue]"))

    # search_food_nutrition のテスト
    console.print("\n[yellow]Testing search_food_nutrition tool...[/yellow]")
    try:
        result = await search_food_nutrition("banana")
        console.print(f"Found {len(result['foods'])} results for 'banana'")

        if result["foods"]:
            food = result["foods"][0]
            console.print(f"First result: {food['name']}")
            console.print(f"Nutrition per 100g: {food['nutrition_per_100g']}")
    except Exception as e:
        console.print(f"[red]Error in search_food_nutrition: {e}[/red]")

    # search_recipes_by_ingredients のテスト
    console.print("\n[yellow]Testing search_recipes_by_ingredients tool...[/yellow]")
    try:
        result = await search_recipes_by_ingredients(
            ["tomato", "pasta"], dietary_restrictions=["vegetarian"]
        )
        console.print(f"Found {len(result['recipes'])} recipes")

        if result["recipes"]:
            recipe = result["recipes"][0]
            console.print(f"First recipe: {recipe['name']}")
            console.print(f"Nutrition: {recipe['nutrition']}")
    except Exception as e:
        console.print(f"[red]Error in search_recipes_by_ingredients: {e}[/red]")


async def test_cache_functionality():
    """キャッシュ機能のテスト"""
    console.print(Panel("[bold blue]キャッシュ機能テスト[/bold blue]"))

    client = FatSecretClient()

    # 同じ検索を2回実行して速度を比較
    search_term = "oatmeal"

    console.print(f"[yellow]First search for '{search_term}'...[/yellow]")
    import time

    start_time = time.time()
    foods1 = await client.search_food(search_term, max_results=2)
    first_time = time.time() - start_time
    console.print(f"First search took: {first_time:.2f}s")

    console.print(
        f"\n[yellow]Second search for '{search_term}' (should be cached)...[/yellow]"
    )
    start_time = time.time()
    foods2 = await client.search_food(search_term, max_results=2)
    second_time = time.time() - start_time
    console.print(f"Second search took: {second_time:.2f}s")

    if second_time < first_time * 0.5:
        console.print("[green]✓ Cache is working! Second search was faster.[/green]")
    else:
        console.print("[yellow]⚠ Cache may not be working as expected.[/yellow]")

    # キャッシュをクリア
    console.print("\n[yellow]Clearing cache...[/yellow]")
    client.clear_cache()

    console.print("[yellow]Third search after cache clear...[/yellow]")
    start_time = time.time()
    foods3 = await client.search_food(search_term, max_results=2)
    third_time = time.time() - start_time
    console.print(f"Third search took: {third_time:.2f}s")


async def main():
    """メインテスト関数"""
    console.print(Panel("[bold green]FatSecret API Client Test Suite[/bold green]"))

    # API認証情報の確認
    if not os.getenv("FATSECRET_CONSUMER_KEY") or not os.getenv(
        "FATSECRET_CONSUMER_SECRET"
    ):
        console.print(
            "[red]Error: FatSecret API credentials not found in environment variables![/red]"
        )
        console.print(
            "Please set FATSECRET_CONSUMER_KEY and FATSECRET_CONSUMER_SECRET in .env file"
        )
        return

    console.print("[green]✓ API credentials found[/green]")

    # 各テストを実行
    tests = [
        ("Food Search", test_food_search),
        ("Recipe Search", test_recipe_search),
        ("Tool Functions", test_tool_functions),
        ("Cache Functionality", test_cache_functionality),
    ]

    for test_name, test_func in tests:
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Running: {test_name}[/bold]")
        console.print(f"{'=' * 60}")

        try:
            await test_func()
            console.print(f"\n[green]✓ {test_name} completed[/green]")
        except Exception as e:
            console.print(f"\n[red]✗ {test_name} failed: {e}[/red]")
            import traceback

            traceback.print_exc()

    console.print(Panel("[bold green]All tests completed![/bold green]"))


if __name__ == "__main__":
    asyncio.run(main())
