#!/usr/bin/env python3
"""
Food-to-Ingredient Mapping Database

Maps specific food items to their constituent ingredients with weight percentages.
Based on USDA food composition data, recipe analysis, and food science literature.

Each food is broken down into base ingredients with their relative proportions
to enable accurate density calculations using weighted averages.
"""

from typing import Dict, List, Tuple, Optional
import logging


class FoodIngredientMapper:
    """
    Maps food items to their constituent ingredients with proportions.

    Each mapping includes:
    - ingredient_name: matches keys in ingredient_density_database.py
    - proportion: decimal percentage (0.0 to 1.0) of the ingredient by weight
    """

    def __init__(self):
        """Initialize the food-to-ingredient mapping database."""
        self._food_mappings = self._load_food_mappings()
        logging.info(f"Loaded ingredient mappings for {len(self._food_mappings)} food items")

    def _load_food_mappings(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Load food-to-ingredient mappings for all 244 specific food classes.

        Based on the food_class_names.py list and research from USDA FoodData Central,
        recipe analysis, and food science literature.
        """

        # Import the complete 244-class mappings
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from complete_244_food_mapping import COMPLETE_244_MAPPINGS
        mappings = COMPLETE_244_MAPPINGS.copy()

        # Validate that all proportions sum to approximately 1.0
        for food_name, ingredients in mappings.items():
            total_proportion = sum(prop for _, prop in ingredients)
            if abs(total_proportion - 1.0) > 0.01:  # Allow 1% tolerance
                logging.warning(f"Food '{food_name}' proportions sum to {total_proportion:.3f}, not 1.0")

        return mappings

    def get_food_ingredients(self, food_name: str) -> Optional[List[Tuple[str, float]]]:
        """
        Get ingredient breakdown for a food item.

        Args:
            food_name: Name of the food item (case-insensitive)

        Returns:
            List of (ingredient_name, proportion) tuples or None if not found
        """
        food_name = food_name.lower().strip().replace(" ", "_")
        return self._food_mappings.get(food_name)

    def get_all_foods(self) -> List[str]:
        """Get list of all mapped food items."""
        return list(self._food_mappings.keys())

    def get_foods_by_cuisine(self, cuisine: str) -> List[str]:
        """Get foods filtered by cuisine type."""
        cuisine_keywords = {
            "asian": ["laksa", "ramen", "pho", "pad_thai", "lo_mein", "biryani", "curry", "sushi", "teriyaki", "tempura", "miso"],
            "italian": ["spaghetti", "fettuccine", "lasagna", "carbonara", "pizza", "risotto"],
            "mexican": ["tacos", "burrito", "quesadilla", "enchiladas"],
            "indian": ["curry", "dal", "naan", "samosa", "biryani"],
            "chinese": ["sweet_and_sour", "kung_pao", "mapo_tofu", "dim_sum", "lo_mein"],
            "japanese": ["sushi", "tempura", "teriyaki", "miso", "ramen"],
            "american": ["hamburger", "cheeseburger", "sandwich", "pizza", "chicken_wings"],
            "seafood": ["fish_and_chips", "salmon", "shrimp", "crab", "sushi"],
            "breakfast": ["pancakes", "french_toast", "omelette", "waffles"],
            "dessert": ["chocolate_cake", "ice_cream", "apple_pie", "cheesecake"]
        }

        keywords = cuisine_keywords.get(cuisine.lower(), [])
        if not keywords:
            return []

        matching_foods = []
        for food_name in self._food_mappings.keys():
            if any(keyword in food_name for keyword in keywords):
                matching_foods.append(food_name)

        return matching_foods

    def add_custom_food_mapping(self, food_name: str, ingredients: List[Tuple[str, float]]):
        """
        Add a custom food-to-ingredient mapping.

        Args:
            food_name: Name of the food item
            ingredients: List of (ingredient_name, proportion) tuples
        """
        # Validate proportions sum to 1.0
        total_proportion = sum(prop for _, prop in ingredients)
        if abs(total_proportion - 1.0) > 0.01:
            raise ValueError(f"Ingredient proportions must sum to 1.0, got {total_proportion:.3f}")

        food_key = food_name.lower().strip().replace(" ", "_")
        self._food_mappings[food_key] = ingredients
        logging.info(f"Added custom food mapping: {food_name}")

    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about the food mapping database."""
        stats = {
            "total_foods": len(self._food_mappings),
            "single_ingredient_foods": 0,
            "complex_foods": 0,
            "max_ingredients": 0,
            "avg_ingredients": 0
        }

        ingredient_counts = []
        for ingredients in self._food_mappings.values():
            count = len(ingredients)
            ingredient_counts.append(count)

            if count == 1:
                stats["single_ingredient_foods"] += 1
            else:
                stats["complex_foods"] += 1

            if count > stats["max_ingredients"]:
                stats["max_ingredients"] = count

        if ingredient_counts:
            stats["avg_ingredients"] = round(sum(ingredient_counts) / len(ingredient_counts), 1)

        return stats


# Global instance
_food_mapper = None

def get_food_ingredient_mapper() -> FoodIngredientMapper:
    """Get global food ingredient mapper instance."""
    global _food_mapper
    if _food_mapper is None:
        _food_mapper = FoodIngredientMapper()
    return _food_mapper