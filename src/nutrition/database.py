#!/usr/bin/env python3
"""
Nutrition Database

Provides nutritional information for food classification categories.
Includes both broad categories (7 Swiss classes) and specific food items.
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass
import logging


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
            )
        }
    
    def _load_specific_class_nutrition(self) -> Dict[str, NutritionInfo]:
        """Load nutritional values for specific food items."""
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
        
        Args:
            food_key: Food identifier (broad class or specific item name)
            
        Returns:
            NutritionInfo object with nutritional data
        """
        food_key = food_key.lower().strip()
        
        # Try specific classification first
        if food_key in self._specific_class_nutrition:
            return self._specific_class_nutrition[food_key]
        
        # Fall back to broad classification
        if food_key in self._broad_class_nutrition:
            return self._broad_class_nutrition[food_key]
        
        # Default to "other" if not found
        logging.warning(f"No nutrition data found for '{food_key}', using 'other' category")
        return self._broad_class_nutrition["other"]
    
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
    
    def get_nutrition_summary(self) -> Dict[str, int]:
        """Get summary statistics about the nutrition database."""
        return {
            "broad_classes": len(self._broad_class_nutrition),
            "specific_items": len(self._specific_class_nutrition),
            "total_entries": len(self._broad_class_nutrition) + len(self._specific_class_nutrition)
        }