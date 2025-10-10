#!/usr/bin/env python3
"""
Nutrition Database

Provides nutritional information for food classification categories.
Includes both broad categories (7 Swiss classes) and specific food items.
"""

from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass
import logging

try:
    from .ingredient_density_database import get_ingredient_database
    from .food_ingredient_mapping import get_food_ingredient_mapper
    INGREDIENT_SYSTEM_AVAILABLE = True
except ImportError:
    logging.warning("Ingredient-based density system not available, using fallback")
    INGREDIENT_SYSTEM_AVAILABLE = False


@dataclass
class NutritionInfo:
    """Nutritional information per 100g of food."""
    
    calories_per_100g: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    density_g_ml: float  # Approximate density for volume-to-weight conversion
    
    def __post_init__(self):
        """Validate nutritional data."""
        if self.calories_per_100g < 0:
            raise ValueError("Calories cannot be negative")
        if self.density_g_ml <= 0:
            raise ValueError("Density must be positive")


class NutritionDatabase:
    """
    Database of nutritional information for food categories and specific items.
    
    Supports both broad classifications (fruit, vegetable, etc.) and specific
    food items (apple, broccoli, etc.) for when friend's classifier is available.
    """
    
    def __init__(self):
        """Initialize nutrition database with food categories and specific items."""
        self._broad_class_nutrition = self._load_broad_class_nutrition()
        self._specific_class_nutrition = self._load_specific_class_nutrition()
        
        logging.info(f"Loaded nutrition data for {len(self._broad_class_nutrition)} broad classes "
                    f"and {len(self._specific_class_nutrition)} specific items")
    
    def _load_broad_class_nutrition(self) -> Dict[str, NutritionInfo]:
        """Load average nutritional values for broad food categories."""
        return {
            "fruit": NutritionInfo(
                calories_per_100g=52.0,
                protein_g=0.8,
                carbs_g=13.0,
                fat_g=0.2,
                fiber_g=2.4,
                density_g_ml=0.85  # Average fruit density
            ),
            "vegetable": NutritionInfo(
                calories_per_100g=25.0,
                protein_g=2.0,
                carbs_g=5.0,
                fat_g=0.3,
                fiber_g=3.0,
                density_g_ml=0.90  # Average vegetable density
            ),
            "carbohydrate": NutritionInfo(
                calories_per_100g=130.0,
                protein_g=2.7,
                carbs_g=28.0,
                fat_g=0.3,
                fiber_g=2.8,
                density_g_ml=1.20  # Rice, bread, pasta average
            ),
            "protein": NutritionInfo(
                calories_per_100g=165.0,
                protein_g=25.0,
                carbs_g=0.0,
                fat_g=6.5,
                fiber_g=0.0,
                density_g_ml=1.05  # Meat, fish average
            ),
            "dairy": NutritionInfo(
                calories_per_100g=150.0,
                protein_g=8.0,
                carbs_g=12.0,
                fat_g=8.5,
                fiber_g=0.0,
                density_g_ml=1.03  # Milk, cheese average
            ),
            "fat": NutritionInfo(
                calories_per_100g=720.0,
                protein_g=0.1,
                carbs_g=0.1,
                fat_g=81.0,
                fiber_g=0.0,
                density_g_ml=0.92  # Oils, butter average
            ),
            "other": NutritionInfo(
                calories_per_100g=200.0,
                protein_g=5.0,
                carbs_g=30.0,
                fat_g=7.0,
                fiber_g=2.0,
                density_g_ml=1.00  # Generic processed foods
            ),
            # 4-class model mappings
            "carb": NutritionInfo(  # Maps to carbohydrate
                calories_per_100g=130.0,
                protein_g=2.7,
                carbs_g=28.0,
                fat_g=0.3,
                fiber_g=2.8,
                density_g_ml=1.20  # Rice, bread, pasta average
            ),
            "meat": NutritionInfo(  # Maps to protein
                calories_per_100g=165.0,
                protein_g=25.0,
                carbs_g=0.0,
                fat_g=6.5,
                fiber_g=0.0,
                density_g_ml=1.05  # Meat, fish average
            ),
            "others": NutritionInfo(  # Maps to other/processed
                calories_per_100g=200.0,
                protein_g=5.0,
                carbs_g=30.0,
                fat_g=7.0,
                fiber_g=2.0,
                density_g_ml=1.00  # Generic processed foods
            )
        }
    
    def _load_specific_class_nutrition(self) -> Dict[str, NutritionInfo]:
        """Load nutritional values for all 233 specific food items from ViT classifier."""
        # Load the complete nutrition mappings
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from complete_233_nutrition_mapping import COMPLETE_233_NUTRITION
            logging.info(f"Successfully loaded {len(COMPLETE_233_NUTRITION)} specific nutrition entries from complete mapping")
            return COMPLETE_233_NUTRITION.copy()
        except ImportError as e:
            # Fallback to basic nutrition data if complete mapping not available
            logging.warning(f"Complete 233-class nutrition mapping not found ({e}), using basic fallback")
            return self._load_basic_specific_nutrition()

    def _load_basic_specific_nutrition(self) -> Dict[str, NutritionInfo]:
        """Fallback basic nutritional values for specific food items."""
        return {
            # Fruits
            "apple": NutritionInfo(52, 0.3, 14, 0.2, 2.4, 0.83),
            "banana": NutritionInfo(89, 1.1, 23, 0.3, 2.6, 0.90),
            "orange": NutritionInfo(47, 0.9, 12, 0.1, 2.4, 0.87),
            "grapes": NutritionInfo(69, 0.7, 18, 0.2, 0.9, 0.80),
            "strawberry": NutritionInfo(32, 0.7, 8, 0.3, 2.0, 0.92),
            "pear": NutritionInfo(57, 0.4, 15, 0.1, 3.1, 0.84),

            # Vegetables
            "broccoli": NutritionInfo(34, 2.8, 7, 0.4, 2.6, 0.89),
            "carrot": NutritionInfo(41, 0.9, 10, 0.2, 2.8, 0.88),
            "tomato": NutritionInfo(18, 0.9, 4, 0.2, 1.2, 0.95),
            "cucumber": NutritionInfo(16, 0.7, 4, 0.1, 0.5, 0.96),
            "lettuce": NutritionInfo(15, 1.4, 3, 0.2, 1.3, 0.95),
            "onion": NutritionInfo(40, 1.1, 9, 0.1, 1.7, 0.90),
            "bell_pepper": NutritionInfo(31, 1.0, 7, 0.3, 2.5, 0.92),
            "spinach": NutritionInfo(23, 2.9, 4, 0.4, 2.2, 0.91),

            # Carbohydrates
            "rice": NutritionInfo(130, 2.7, 28, 0.3, 0.4, 1.25),
            "bread": NutritionInfo(265, 9.0, 49, 3.2, 2.7, 0.40),
            "pasta": NutritionInfo(131, 5.0, 25, 1.1, 1.8, 1.20),
            "potato": NutritionInfo(77, 2.0, 17, 0.1, 2.2, 1.08),
            "sweet_potato": NutritionInfo(86, 1.6, 20, 0.1, 3.0, 1.06),
            "quinoa": NutritionInfo(120, 4.4, 22, 1.9, 2.8, 1.30),
            "oats": NutritionInfo(68, 2.5, 12, 1.4, 1.7, 0.75),

            # Proteins
            "chicken_breast": NutritionInfo(165, 31, 0, 3.6, 0, 1.06),
            "salmon": NutritionInfo(208, 25, 0, 12, 0, 1.05),
            "beef": NutritionInfo(250, 26, 0, 15, 0, 1.04),
            "pork": NutritionInfo(242, 27, 0, 14, 0, 1.03),
            "tuna": NutritionInfo(144, 30, 0, 1, 0, 1.08),
            "egg": NutritionInfo(155, 13, 1.1, 11, 0, 1.03),
            "tofu": NutritionInfo(76, 8.1, 1.9, 4.8, 0.3, 1.02),
            "beans": NutritionInfo(127, 8.7, 23, 0.5, 6.4, 1.15),

            # Dairy
            "milk": NutritionInfo(42, 3.4, 5.0, 1.0, 0, 1.03),
            "cheese": NutritionInfo(356, 25, 2.4, 29, 0, 1.15),
            "yogurt": NutritionInfo(59, 10, 3.6, 0.4, 0, 1.04),
            "butter": NutritionInfo(717, 0.9, 0.1, 81, 0, 0.91),

            # Fats/Oils
            "olive_oil": NutritionInfo(884, 0, 0, 100, 0, 0.92),
            "avocado": NutritionInfo(160, 2.0, 9, 15, 7, 0.96),
            "nuts_mixed": NutritionInfo(607, 20, 16, 54, 7, 0.65),
            "almonds": NutritionInfo(579, 21, 22, 50, 12, 0.64),

            # Other/Processed
            "pizza": NutritionInfo(266, 11, 33, 10, 2.3, 0.85),
            "sandwich": NutritionInfo(250, 12, 30, 10, 3, 0.70),
            "soup": NutritionInfo(50, 2.5, 8, 1.5, 1, 1.00),
            "salad": NutritionInfo(20, 1.5, 4, 0.2, 2, 0.95)
        }
    
    def get_nutrition_info(self, food_key: str) -> NutritionInfo:
        """
        Get nutrition information for a food item.

        Prioritizes specific classifier results over broad segmentation categories.

        Args:
            food_key: Food identifier (specific class name preferred, or broad class)

        Returns:
            NutritionInfo object with nutritional data
        """
        food_key = food_key.lower().strip()

        # PRIORITY 1: Try specific classification first (from 233-class ViT classifier)
        if food_key in self._specific_class_nutrition:
            logging.debug(f"Found specific nutrition data for '{food_key}'")
            return self._specific_class_nutrition[food_key]

        # PRIORITY 2: Try fuzzy matching for specific foods with common variations
        specific_matches = self._fuzzy_match_specific_food(food_key)
        if specific_matches:
            logging.debug(f"Fuzzy matched '{food_key}' to specific food '{specific_matches}'")
            return self._specific_class_nutrition[specific_matches]

        # PRIORITY 3: Fall back to broad classification (7-class segmentation)
        if food_key in self._broad_class_nutrition:
            logging.debug(f"Using broad classification for '{food_key}'")
            return self._broad_class_nutrition[food_key]

        # PRIORITY 4: Default to "other" if not found
        logging.warning(f"No nutrition data found for '{food_key}', using 'other' category")
        return self._broad_class_nutrition["other"]

    def _fuzzy_match_specific_food(self, food_key: str) -> Optional[str]:
        """
        Try to match a food name to specific foods using fuzzy matching.

        Args:
            food_key: Food name to match

        Returns:
            Matched specific food name or None
        """
        # Common synonyms and variations
        synonyms = {
            # Fruits
            "apples": "apple",
            "bananas": "banana",
            "oranges": "orange",
            "strawberries": "strawberry",
            "grapes": "grape",

            # Common food variations
            "rice_cooked": "cooked_white_rice",
            "rice_white": "cooked_white_rice",
            "white_rice": "cooked_white_rice",
            "brown_rice": "cooked_brown_rice",
            "fried_rice": "fried_rice",

            # Bread variations
            "bread_white": "white_bread",
            "bread_whole_wheat": "whole_grain_bread",

            # Chicken variations
            "chicken": "chicken_rice",
            "chicken_breast": "chicken_chop",

            # Noodle variations
            "noodles": "laksa",
            "pasta": "pasta_red_sauce",

            # Generic mappings to specific dishes
            "soup": "chicken_soup",
            "salad": "salad",
        }

        # Direct synonym lookup
        if food_key in synonyms:
            matched = synonyms[food_key]
            if matched in self._specific_class_nutrition:
                return matched

        # Try partial matching for compound food names
        for specific_food in self._specific_class_nutrition.keys():
            if food_key in specific_food or specific_food in food_key:
                # Ensure it's a meaningful match (not too short)
                if len(food_key) >= 3 and len(specific_food) >= 3:
                    return specific_food

        return None
    
    def get_broad_class_nutrition(self, broad_class: str) -> NutritionInfo:
        """Get nutrition info for a broad food class."""
        broad_class = broad_class.lower().strip()
        return self._broad_class_nutrition.get(broad_class, self._broad_class_nutrition["other"])
    
    def get_specific_class_nutrition(self, specific_class: str) -> Optional[NutritionInfo]:
        """Get nutrition info for a specific food item."""
        specific_class = specific_class.lower().strip()
        return self._specific_class_nutrition.get(specific_class)
    
    def get_supported_broad_classes(self) -> list[str]:
        """Get list of supported broad food classes."""
        return list(self._broad_class_nutrition.keys())
    
    def get_supported_specific_classes(self) -> list[str]:
        """Get list of supported specific food items."""
        return list(self._specific_class_nutrition.keys())
    
    def add_custom_nutrition(self, food_name: str, nutrition_info: NutritionInfo, is_specific: bool = True):
        """
        Add custom nutrition information for a food item.
        
        Args:
            food_name: Name of the food item
            nutrition_info: Nutrition information
            is_specific: Whether this is a specific item (True) or broad class (False)
        """
        food_name = food_name.lower().strip()
        
        if is_specific:
            self._specific_class_nutrition[food_name] = nutrition_info
        else:
            self._broad_class_nutrition[food_name] = nutrition_info
        
        logging.info(f"Added custom nutrition data for {food_name}")
    
    def estimate_nutrition_from_macros(self, protein_g: float, carbs_g: float, fat_g: float, 
                                     fiber_g: float = 0.0, density_g_ml: float = 1.0) -> NutritionInfo:
        """
        Estimate nutrition info from basic macronutrient values.
        
        Uses standard caloric values: protein=4 cal/g, carbs=4 cal/g, fat=9 cal/g
        """
        calories = (protein_g * 4.0) + (carbs_g * 4.0) + (fat_g * 9.0)
        
        return NutritionInfo(
            calories_per_100g=calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
            fiber_g=fiber_g,
            density_g_ml=density_g_ml
        )

    def get_ingredient_based_density(self, food_name: str) -> Optional[float]:
        """
        Calculate density using ingredient breakdown and weighted averages.

        Args:
            food_name: Name of the specific food item

        Returns:
            Calculated density in g/ml or None if ingredient data unavailable
        """
        if not INGREDIENT_SYSTEM_AVAILABLE:
            return None

        try:
            # Get ingredient mapping
            mapper = get_food_ingredient_mapper()
            ingredients = mapper.get_food_ingredients(food_name)

            if not ingredients:
                return None

            # Get ingredient density database
            ingredient_db = get_ingredient_database()

            # Calculate weighted average density
            total_weighted_density = 0.0
            total_weight = 0.0

            for ingredient_name, proportion in ingredients:
                ingredient_density = ingredient_db.get_ingredient_density(ingredient_name)

                if ingredient_density is None:
                    logging.warning(f"No density found for ingredient '{ingredient_name}' in food '{food_name}'")
                    continue

                total_weighted_density += ingredient_density * proportion
                total_weight += proportion

            # Return weighted average if we found at least some ingredients
            if total_weight > 0:
                calculated_density = total_weighted_density / total_weight
                logging.debug(f"Calculated ingredient-based density for '{food_name}': {calculated_density:.3f} g/ml")
                return calculated_density
            else:
                logging.warning(f"No valid ingredients found for density calculation of '{food_name}'")
                return None

        except Exception as e:
            logging.error(f"Error calculating ingredient-based density for '{food_name}': {e}")
            return None

    def get_enhanced_nutrition_info(self, food_key: str) -> NutritionInfo:
        """
        Get nutrition information with enhanced ingredient-based density calculation.

        Args:
            food_key: Food identifier (broad class or specific item name)

        Returns:
            NutritionInfo object with enhanced density calculation
        """
        # Get base nutrition info using existing method
        nutrition_info = self.get_nutrition_info(food_key)

        # Try to enhance with ingredient-based density
        ingredient_density = self.get_ingredient_based_density(food_key)

        if ingredient_density is not None:
            # Create new NutritionInfo with enhanced density
            enhanced_info = NutritionInfo(
                calories_per_100g=nutrition_info.calories_per_100g,
                protein_g=nutrition_info.protein_g,
                carbs_g=nutrition_info.carbs_g,
                fat_g=nutrition_info.fat_g,
                fiber_g=nutrition_info.fiber_g,
                density_g_ml=ingredient_density  # Use ingredient-based density
            )
            logging.debug(f"Enhanced density for '{food_key}': {ingredient_density:.3f} g/ml "
                         f"(was {nutrition_info.density_g_ml:.3f} g/ml)")
            return enhanced_info
        else:
            # Fall back to original density
            return nutrition_info

    def get_ingredient_breakdown_info(self, food_name: str) -> Optional[Dict[str, float]]:
        """
        Get detailed ingredient breakdown with individual densities.

        Args:
            food_name: Name of the specific food item

        Returns:
            Dictionary mapping ingredient names to their densities
        """
        if not INGREDIENT_SYSTEM_AVAILABLE:
            return None

        try:
            mapper = get_food_ingredient_mapper()
            ingredients = mapper.get_food_ingredients(food_name)

            if not ingredients:
                return None

            ingredient_db = get_ingredient_database()
            breakdown = {}

            for ingredient_name, proportion in ingredients:
                ingredient_density = ingredient_db.get_ingredient_density(ingredient_name)
                if ingredient_density is not None:
                    breakdown[ingredient_name] = {
                        'proportion': proportion,
                        'density_g_ml': ingredient_density
                    }

            return breakdown

        except Exception as e:
            logging.error(f"Error getting ingredient breakdown for '{food_name}': {e}")
            return None
    
    def get_nutrition_summary(self) -> Dict[str, int]:
        """Get summary statistics about the nutrition database."""
        return {
            "broad_classes": len(self._broad_class_nutrition),
            "specific_items": len(self._specific_class_nutrition),
            "total_entries": len(self._broad_class_nutrition) + len(self._specific_class_nutrition)
        }