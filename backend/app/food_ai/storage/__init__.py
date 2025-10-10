#!/usr/bin/env python3
"""
Storage Module

Provides data storage and management for nutrition analysis including
nutrition database, food class mappings, and external API integration.
"""

from .nutrition_database import NutritionDatabase, NutritionInfo
from .food_class_names import FOOD_CLASS_NAMES

# Nutritionix API is optional; import guarded to avoid hard dependency in API runtime
try:
    from .nutritionix_api import NutritionixAPI  # type: ignore
except Exception:  # pragma: no cover
    NutritionixAPI = None  # Optional

__all__ = ['NutritionDatabase', 'NutritionInfo', 'FOOD_CLASS_NAMES', 'NutritionixAPI']
