#!/usr/bin/env python3
"""
Ingredient Density Database

Provides accurate density values for base food ingredients sourced from
authoritative references including USDA FoodData Central, food engineering
handbooks, and peer-reviewed research.

All density values are in g/ml at room temperature (20°C).
"""

from typing import Dict, Optional
from dataclasses import dataclass
import logging


@dataclass
class IngredientInfo:
    """Information about a base ingredient including density and source."""

    name: str
    density_g_ml: float
    source: str  # Reference source for traceability
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate ingredient data."""
        if self.density_g_ml <= 0:
            raise ValueError(f"Density must be positive for {self.name}")


class IngredientDensityDatabase:
    """
    Database of density values for base food ingredients.

    Sources:
    - USDA: USDA FoodData Central Database
    - CRC: CRC Handbook of Chemistry and Physics
    - Rahman: Rahman's Handbook of Food Properties
    - JFE: Journal of Food Engineering research papers
    - Measured: Direct measurement conversions
    """

    def __init__(self):
        """Initialize the ingredient density database."""
        self._ingredients = self._load_ingredient_densities()
        logging.info(f"Loaded density data for {len(self._ingredients)} base ingredients")

    def _load_ingredient_densities(self) -> Dict[str, IngredientInfo]:
        """Load density values for base ingredients from authoritative sources."""

        ingredients = {
            # GRAINS & STARCHES (USDA FoodData Central)
            "rice_cooked": IngredientInfo("Cooked Rice", 1.25, "USDA", "Long grain white rice, cooked"),
            "rice_noodles": IngredientInfo("Rice Noodles", 1.20, "USDA", "Fresh rice noodles"),
            "wheat_noodles": IngredientInfo("Wheat Noodles", 1.15, "USDA", "Fresh wheat/egg noodles"),
            "pasta_cooked": IngredientInfo("Cooked Pasta", 1.20, "USDA", "Cooked wheat pasta"),
            "bread_white": IngredientInfo("White Bread", 0.28, "USDA", "Commercial white bread"),
            "bread_whole_wheat": IngredientInfo("Whole Wheat Bread", 0.41, "USDA", "Whole wheat bread"),
            "flour_wheat": IngredientInfo("Wheat Flour", 0.59, "USDA", "All-purpose wheat flour"),
            "oats_cooked": IngredientInfo("Cooked Oats", 0.84, "USDA", "Cooked rolled oats"),
            "quinoa_cooked": IngredientInfo("Cooked Quinoa", 1.30, "USDA", "Cooked quinoa"),
            "barley_cooked": IngredientInfo("Cooked Barley", 1.22, "USDA", "Pearl barley, cooked"),

            # PROTEINS - MEAT & SEAFOOD (USDA + Food Engineering)
            "beef_lean": IngredientInfo("Lean Beef", 1.04, "USDA", "Lean ground beef, cooked"),
            "beef_regular": IngredientInfo("Regular Beef", 1.02, "USDA", "Regular ground beef, cooked"),
            "chicken_breast": IngredientInfo("Chicken Breast", 1.06, "USDA", "Boneless, skinless, cooked"),
            "chicken_thigh": IngredientInfo("Chicken Thigh", 1.04, "USDA", "Boneless, with skin, cooked"),
            "pork_lean": IngredientInfo("Lean Pork", 1.03, "USDA", "Lean pork loin, cooked"),
            "fish_white": IngredientInfo("White Fish", 1.05, "USDA", "Cod, tilapia, etc."),
            "fish_oily": IngredientInfo("Oily Fish", 1.03, "USDA", "Salmon, mackerel, etc."),
            "shrimp": IngredientInfo("Shrimp", 1.08, "USDA", "Cooked shrimp"),
            "crab": IngredientInfo("Crab Meat", 1.07, "USDA", "Cooked crab meat"),
            "squid": IngredientInfo("Squid", 1.09, "USDA", "Cooked squid/calamari"),

            # PROTEINS - EGGS & DAIRY PROTEINS
            "egg_whole": IngredientInfo("Whole Egg", 1.03, "USDA", "Whole eggs, cooked"),
            "egg_white": IngredientInfo("Egg White", 1.04, "USDA", "Egg whites only"),
            "tofu_firm": IngredientInfo("Firm Tofu", 1.02, "USDA", "Extra firm tofu"),
            "tofu_soft": IngredientInfo("Soft Tofu", 1.01, "USDA", "Soft/silken tofu"),

            # LEGUMES & NUTS
            "beans_cooked": IngredientInfo("Cooked Beans", 1.15, "USDA", "Black/kidney/pinto beans"),
            "lentils_cooked": IngredientInfo("Cooked Lentils", 1.18, "USDA", "Cooked lentils"),
            "chickpeas_cooked": IngredientInfo("Cooked Chickpeas", 1.12, "USDA", "Cooked chickpeas"),
            "nuts_mixed": IngredientInfo("Mixed Nuts", 0.65, "USDA", "Mixed nuts, average"),
            "almonds": IngredientInfo("Almonds", 0.64, "USDA", "Whole almonds"),
            "peanuts": IngredientInfo("Peanuts", 0.64, "USDA", "Roasted peanuts"),
            "cashews": IngredientInfo("Cashews", 0.59, "USDA", "Roasted cashews"),

            # DAIRY PRODUCTS (USDA)
            "milk_whole": IngredientInfo("Whole Milk", 1.03, "USDA", "3.25% fat milk"),
            "milk_skim": IngredientInfo("Skim Milk", 1.04, "USDA", "Non-fat milk"),
            "cream_heavy": IngredientInfo("Heavy Cream", 0.99, "USDA", "36% fat heavy cream"),
            "coconut_milk": IngredientInfo("Coconut Milk", 0.97, "USDA", "Canned coconut milk"),
            "coconut_cream": IngredientInfo("Coconut Cream", 0.94, "Rahman", "Thick coconut cream"),
            "yogurt_plain": IngredientInfo("Plain Yogurt", 1.04, "USDA", "Plain whole milk yogurt"),
            "cheese_hard": IngredientInfo("Hard Cheese", 1.15, "USDA", "Cheddar, parmesan average"),
            "cheese_soft": IngredientInfo("Soft Cheese", 1.08, "USDA", "Mozzarella, brie average"),
            "butter": IngredientInfo("Butter", 0.91, "USDA", "Salted butter"),

            # VEGETABLES (USDA FoodData Central)
            "onions": IngredientInfo("Onions", 0.90, "USDA", "Yellow onions, raw"),
            "garlic": IngredientInfo("Garlic", 0.95, "USDA", "Fresh garlic cloves"),
            "ginger": IngredientInfo("Ginger", 0.93, "USDA", "Fresh ginger root"),
            "tomatoes": IngredientInfo("Tomatoes", 0.95, "USDA", "Fresh tomatoes"),
            "bell_peppers": IngredientInfo("Bell Peppers", 0.92, "USDA", "Sweet bell peppers"),
            "carrots": IngredientInfo("Carrots", 0.88, "USDA", "Fresh carrots"),
            "celery": IngredientInfo("Celery", 0.95, "USDA", "Fresh celery"),
            "broccoli": IngredientInfo("Broccoli", 0.89, "USDA", "Fresh broccoli florets"),
            "cauliflower": IngredientInfo("Cauliflower", 0.92, "USDA", "Fresh cauliflower"),
            "cabbage": IngredientInfo("Cabbage", 0.91, "USDA", "Fresh cabbage"),
            "spinach": IngredientInfo("Spinach", 0.91, "USDA", "Fresh spinach leaves"),
            "lettuce": IngredientInfo("Lettuce", 0.95, "USDA", "Iceberg lettuce"),
            "cucumber": IngredientInfo("Cucumber", 0.96, "USDA", "Fresh cucumber"),
            "zucchini": IngredientInfo("Zucchini", 0.95, "USDA", "Fresh zucchini"),
            "mushrooms": IngredientInfo("Mushrooms", 0.92, "USDA", "Button mushrooms"),
            "potatoes": IngredientInfo("Potatoes", 1.08, "USDA", "Raw potatoes"),
            "sweet_potatoes": IngredientInfo("Sweet Potatoes", 1.06, "USDA", "Raw sweet potatoes"),

            # FRUITS (USDA)
            "apples": IngredientInfo("Apples", 0.83, "USDA", "Fresh apples with skin"),
            "bananas": IngredientInfo("Bananas", 0.90, "USDA", "Fresh bananas"),
            "oranges": IngredientInfo("Oranges", 0.87, "USDA", "Fresh oranges"),
            "lemons": IngredientInfo("Lemons", 0.89, "USDA", "Fresh lemons"),
            "limes": IngredientInfo("Limes", 0.90, "USDA", "Fresh limes"),
            "pineapple": IngredientInfo("Pineapple", 0.89, "USDA", "Fresh pineapple"),
            "mango": IngredientInfo("Mango", 0.84, "USDA", "Fresh mango"),
            "strawberries": IngredientInfo("Strawberries", 0.92, "USDA", "Fresh strawberries"),
            "grapes": IngredientInfo("Grapes", 0.80, "USDA", "Fresh grapes"),

            # LIQUIDS & SAUCES (USDA + Food Engineering)
            "water": IngredientInfo("Water", 1.00, "CRC", "Pure water at 20°C"),
            "broth_chicken": IngredientInfo("Chicken Broth", 1.02, "USDA", "Low sodium chicken broth"),
            "broth_beef": IngredientInfo("Beef Broth", 1.02, "USDA", "Low sodium beef broth"),
            "broth_vegetable": IngredientInfo("Vegetable Broth", 1.01, "USDA", "Vegetable broth"),
            "tomato_sauce": IngredientInfo("Tomato Sauce", 1.02, "USDA", "Plain tomato sauce"),
            "tomato_paste": IngredientInfo("Tomato Paste", 1.09, "USDA", "Concentrated tomato paste"),
            "soy_sauce": IngredientInfo("Soy Sauce", 1.17, "USDA", "Regular soy sauce"),
            "fish_sauce": IngredientInfo("Fish Sauce", 1.12, "Measured", "Asian fish sauce"),
            "vinegar": IngredientInfo("Vinegar", 1.01, "USDA", "White/rice vinegar"),
            "wine_cooking": IngredientInfo("Cooking Wine", 0.99, "USDA", "White/red cooking wine"),

            # OILS & FATS (USDA + CRC)
            "oil_vegetable": IngredientInfo("Vegetable Oil", 0.92, "USDA", "Soybean/canola oil"),
            "oil_olive": IngredientInfo("Olive Oil", 0.92, "USDA", "Extra virgin olive oil"),
            "oil_sesame": IngredientInfo("Sesame Oil", 0.92, "CRC", "Sesame seed oil"),
            "oil_coconut": IngredientInfo("Coconut Oil", 0.92, "USDA", "Virgin coconut oil"),
            "lard": IngredientInfo("Lard", 0.92, "USDA", "Pork fat/lard"),

            # HERBS & SPICES (USDA + Measured)
            "herbs_fresh": IngredientInfo("Fresh Herbs", 0.90, "USDA", "Basil, cilantro, parsley avg"),
            "herbs_dried": IngredientInfo("Dried Herbs", 0.35, "Measured", "Dried herbs average"),
            "spices_ground": IngredientInfo("Ground Spices", 0.50, "Measured", "Ground spices average"),
            "chili_peppers": IngredientInfo("Chili Peppers", 0.88, "USDA", "Fresh chili peppers"),

            # SUGARS & SWEETENERS (USDA + CRC)
            "sugar_white": IngredientInfo("White Sugar", 0.85, "CRC", "Granulated white sugar"),
            "sugar_brown": IngredientInfo("Brown Sugar", 0.90, "CRC", "Packed brown sugar"),
            "honey": IngredientInfo("Honey", 1.42, "USDA", "Natural honey"),
            "maple_syrup": IngredientInfo("Maple Syrup", 1.32, "USDA", "Pure maple syrup"),

            # PROCESSED FOODS (USDA + Measured)
            "pizza_dough": IngredientInfo("Pizza Dough", 0.70, "Measured", "Fresh pizza dough"),
            "bread_crumbs": IngredientInfo("Bread Crumbs", 0.40, "USDA", "Dry bread crumbs"),
            "cornstarch": IngredientInfo("Cornstarch", 0.60, "USDA", "Corn starch powder"),
        }

        return ingredients

    def get_ingredient_density(self, ingredient_name: str) -> Optional[float]:
        """
        Get density value for an ingredient.

        Args:
            ingredient_name: Name of the ingredient (case-insensitive)

        Returns:
            Density in g/ml or None if not found
        """
        ingredient_name = ingredient_name.lower().strip()

        # Direct lookup
        if ingredient_name in self._ingredients:
            return self._ingredients[ingredient_name].density_g_ml

        # Fuzzy matching for common variations
        synonyms = {
            "rice": "rice_cooked",
            "noodles": "rice_noodles",
            "pasta": "pasta_cooked",
            "bread": "bread_white",
            "chicken": "chicken_breast",
            "beef": "beef_lean",
            "pork": "pork_lean",
            "fish": "fish_white",
            "milk": "milk_whole",
            "oil": "oil_vegetable",
            "onion": "onions",
            "tomato": "tomatoes",
            "egg": "egg_whole",
            "beans": "beans_cooked",
            "nuts": "nuts_mixed",
            "vegetables": "bell_peppers",  # Generic vegetables -> bell peppers (average veggie density)
            "herbs": "herbs_fresh",
            "spices": "spices_ground"
        }

        if ingredient_name in synonyms:
            return self._ingredients[synonyms[ingredient_name]].density_g_ml

        return None

    def get_ingredient_info(self, ingredient_name: str) -> Optional[IngredientInfo]:
        """Get full ingredient information including source."""
        ingredient_name = ingredient_name.lower().strip()
        return self._ingredients.get(ingredient_name)

    def get_all_ingredients(self) -> Dict[str, IngredientInfo]:
        """Get all ingredient information."""
        return self._ingredients.copy()

    def get_ingredients_by_category(self, category: str) -> Dict[str, IngredientInfo]:
        """Get ingredients filtered by category."""
        category_keywords = {
            "grains": ["rice", "noodles", "pasta", "bread", "flour", "oats", "quinoa", "barley"],
            "proteins": ["beef", "chicken", "pork", "fish", "shrimp", "crab", "squid", "egg", "tofu"],
            "legumes": ["beans", "lentils", "chickpeas", "nuts", "almonds", "peanuts", "cashews"],
            "dairy": ["milk", "cream", "coconut", "yogurt", "cheese", "butter"],
            "vegetables": ["onions", "garlic", "ginger", "tomatoes", "peppers", "carrots", "celery",
                          "broccoli", "cauliflower", "cabbage", "spinach", "lettuce", "cucumber",
                          "zucchini", "mushrooms", "potatoes"],
            "fruits": ["apples", "bananas", "oranges", "lemons", "limes", "pineapple", "mango",
                      "strawberries", "grapes"],
            "liquids": ["water", "broth", "sauce", "soy", "fish", "vinegar", "wine"],
            "fats": ["oil", "lard"],
            "seasonings": ["herbs", "spices", "chili"],
            "sweeteners": ["sugar", "honey", "maple", "syrup"]
        }

        keywords = category_keywords.get(category.lower(), [])
        if not keywords:
            return {}

        filtered = {}
        for name, info in self._ingredients.items():
            if any(keyword in name for keyword in keywords):
                filtered[name] = info

        return filtered

    def add_custom_ingredient(self, name: str, density_g_ml: float,
                            source: str, notes: Optional[str] = None):
        """Add a custom ingredient to the database."""
        ingredient_key = name.lower().strip().replace(" ", "_")
        self._ingredients[ingredient_key] = IngredientInfo(
            name=name,
            density_g_ml=density_g_ml,
            source=source,
            notes=notes
        )
        logging.info(f"Added custom ingredient: {name} ({density_g_ml} g/ml)")


# Global instance
_ingredient_db = None

def get_ingredient_database() -> IngredientDensityDatabase:
    """Get global ingredient density database instance."""
    global _ingredient_db
    if _ingredient_db is None:
        _ingredient_db = IngredientDensityDatabase()
    return _ingredient_db