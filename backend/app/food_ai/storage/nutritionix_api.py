#!/usr/bin/env python3
"""
Nutritionix API Integration

Provides access to Nutritionix's comprehensive nutrition database for real-time
food nutrition data retrieval and caching.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import requests
from pathlib import Path


@dataclass
class NutritionixFood:
    """Nutritionix food item with detailed nutrition information."""
    
    food_name: str
    brand_name: Optional[str]
    serving_qty: float
    serving_unit: str
    serving_weight_grams: float
    nf_calories: float
    nf_total_fat: float
    nf_saturated_fat: Optional[float]
    nf_cholesterol: Optional[float]
    nf_sodium: Optional[float]
    nf_total_carbohydrate: float
    nf_dietary_fiber: Optional[float]
    nf_sugars: Optional[float]
    nf_protein: float
    nf_potassium: Optional[float]
    
    # Additional fields for our database
    density_g_ml: float = 1.0  # Estimated density
    confidence: float = 1.0    # API confidence score
    
    def to_nutrition_info(self):
        """Convert to our internal NutritionInfo format (per 100g)."""
        from .database import NutritionInfo
        
        # Convert to per-100g values
        factor = 100.0 / max(self.serving_weight_grams, 1.0)
        
        return NutritionInfo(
            calories_per_100g=self.nf_calories * factor,
            protein_g=self.nf_protein * factor,
            carbs_g=self.nf_total_carbohydrate * factor,
            fat_g=self.nf_total_fat * factor,
            fiber_g=(self.nf_dietary_fiber or 0.0) * factor,
            density_g_ml=self.density_g_ml
        )


class NutritionixAPIError(Exception):
    """Custom exception for Nutritionix API errors."""
    pass


class NutritionixAPI:
    """
    Interface to Nutritionix API for real-time nutrition data.
    
    Provides caching, rate limiting, and fallback mechanisms for robust
    nutrition data retrieval.
    """
    
    BASE_URL = "https://trackapi.nutritionix.com/v2"
    
    def __init__(self, app_id: Optional[str] = None, app_key: Optional[str] = None, 
                 cache_dir: Optional[str] = None):
        """
        Initialize Nutritionix API client.
        
        Args:
            app_id: Nutritionix Application ID (or from NUTRITIONIX_APP_ID env var)
            app_key: Nutritionix Application Key (or from NUTRITIONIX_APP_KEY env var)
            cache_dir: Directory for caching API responses
        """
        self.app_id = app_id or os.getenv('NUTRITIONIX_APP_ID')
        self.app_key = app_key or os.getenv('NUTRITIONIX_APP_KEY')
        
        if not self.app_id or not self.app_key:
            logging.warning("Nutritionix API credentials not found. API features will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            
        # Setup caching
        self.cache_dir = Path(cache_dir or "cache/nutritionix")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting (500 requests/hour for free tier)
        self.rate_limit_delay = 7.2  # seconds between requests
        self.last_request_time = 0
        
        # Headers for API requests
        self.headers = {
            'Content-Type': 'application/json',
            'x-app-id': self.app_id,
            'x-app-key': self.app_key
        }
        
        logging.info(f"Nutritionix API initialized (enabled: {self.enabled})")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests."""
        if not self.enabled:
            return
            
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logging.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_path(self, query: str) -> Path:
        """Get cache file path for a query."""
        # Create safe filename from query
        safe_query = "".join(c if c.isalnum() or c in '-_' else '_' for c in query.lower())
        return self.cache_dir / f"{safe_query[:50]}.json"
    
    def _load_from_cache(self, query: str) -> Optional[Dict]:
        """Load cached API response."""
        cache_path = self._get_cache_path(query)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid (24 hours)
                cache_age = time.time() - cached_data.get('timestamp', 0)
                if cache_age < 24 * 3600:  # 24 hours
                    logging.debug(f"Using cached data for query: {query}")
                    return cached_data.get('data')
            except Exception as e:
                logging.warning(f"Failed to load cache for {query}: {e}")
        
        return None
    
    def _save_to_cache(self, query: str, data: Dict):
        """Save API response to cache."""
        cache_path = self._get_cache_path(query)
        try:
            cache_data = {
                'timestamp': time.time(),
                'query': query,
                'data': data
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logging.debug(f"Cached data for query: {query}")
        except Exception as e:
            logging.warning(f"Failed to cache data for {query}: {e}")
    
    def search_food(self, query: str) -> List[NutritionixFood]:
        """
        Search for foods using natural language query.
        
        Args:
            query: Natural language food description (e.g., "1 cup rice", "apple")
            
        Returns:
            List of NutritionixFood objects
        """
        if not self.enabled:
            logging.warning("Nutritionix API not available")
            return []
        
        # Check cache first
        cached_result = self._load_from_cache(query)
        if cached_result:
            return [NutritionixFood(**item) for item in cached_result]
        
        # Make API request
        self._enforce_rate_limit()
        
        try:
            url = f"{self.BASE_URL}/natural/nutrients"
            payload = {"query": query}
            
            logging.debug(f"Nutritionix API request: {query}")
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            foods = data.get('foods', [])
            
            # Convert to our format
            nutritionix_foods = []
            for food_data in foods:
                try:
                    # Estimate density based on food type
                    density = self._estimate_density(food_data.get('food_name', ''))
                    
                    nutritionix_food = NutritionixFood(
                        food_name=food_data.get('food_name', ''),
                        brand_name=food_data.get('brand_name'),
                        serving_qty=food_data.get('serving_qty', 1.0),
                        serving_unit=food_data.get('serving_unit', 'serving'),
                        serving_weight_grams=food_data.get('serving_weight_grams', 100.0),
                        nf_calories=food_data.get('nf_calories', 0.0),
                        nf_total_fat=food_data.get('nf_total_fat', 0.0),
                        nf_saturated_fat=food_data.get('nf_saturated_fat'),
                        nf_cholesterol=food_data.get('nf_cholesterol'),
                        nf_sodium=food_data.get('nf_sodium'),
                        nf_total_carbohydrate=food_data.get('nf_total_carbohydrate', 0.0),
                        nf_dietary_fiber=food_data.get('nf_dietary_fiber'),
                        nf_sugars=food_data.get('nf_sugars'),
                        nf_protein=food_data.get('nf_protein', 0.0),
                        nf_potassium=food_data.get('nf_potassium'),
                        density_g_ml=density,
                        confidence=1.0  # High confidence for API data
                    )
                    nutritionix_foods.append(nutritionix_food)
                    
                except Exception as e:
                    logging.warning(f"Failed to parse food data: {e}")
                    continue
            
            # Cache the results
            cache_data = [asdict(food) for food in nutritionix_foods]
            self._save_to_cache(query, cache_data)
            
            logging.info(f"Retrieved {len(nutritionix_foods)} foods for query: {query}")
            return nutritionix_foods
            
        except requests.RequestException as e:
            logging.error(f"Nutritionix API request failed: {e}")
            raise NutritionixAPIError(f"API request failed: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in Nutritionix API: {e}")
            raise NutritionixAPIError(f"Unexpected error: {e}")
    
    def get_nutrition_by_name(self, food_name: str) -> Optional[NutritionixFood]:
        """
        Get nutrition information for a specific food name.
        
        Args:
            food_name: Name of the food item
            
        Returns:
            NutritionixFood object or None if not found
        """
        foods = self.search_food(food_name)
        if foods:
            # Return the first (most relevant) result
            return foods[0]
        return None
    
    def _estimate_density(self, food_name: str) -> float:
        """
        Estimate density based on food name and type.
        
        Args:
            food_name: Name of the food
            
        Returns:
            Estimated density in g/ml
        """
        food_name = food_name.lower()
        
        # Density estimates based on food categories
        if any(word in food_name for word in ['oil', 'butter', 'margarine']):
            return 0.92
        elif any(word in food_name for word in ['bread', 'cake', 'muffin', 'biscuit']):
            return 0.40
        elif any(word in food_name for word in ['rice', 'quinoa', 'barley']):
            return 1.25
        elif any(word in food_name for word in ['pasta', 'noodle', 'spaghetti']):
            return 1.20
        elif any(word in food_name for word in ['meat', 'chicken', 'beef', 'pork', 'fish']):
            return 1.05
        elif any(word in food_name for word in ['milk', 'yogurt', 'cheese']):
            return 1.03
        elif any(word in food_name for word in ['fruit', 'apple', 'banana', 'orange']):
            return 0.85
        elif any(word in food_name for word in ['vegetable', 'broccoli', 'carrot', 'tomato']):
            return 0.90
        elif any(word in food_name for word in ['nuts', 'almond', 'walnut']):
            return 0.65
        elif any(word in food_name for word in ['soup', 'broth', 'juice']):
            return 1.00
        else:
            return 1.00  # Default density
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and configuration information."""
        return {
            'enabled': self.enabled,
            'has_credentials': bool(self.app_id and self.app_key),
            'cache_dir': str(self.cache_dir),
            'cache_files': len(list(self.cache_dir.glob('*.json'))) if self.cache_dir.exists() else 0,
            'rate_limit_delay': self.rate_limit_delay
        }
    
    def clear_cache(self):
        """Clear all cached API responses."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob('*.json'):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete cache file {cache_file}: {e}")
            logging.info("Nutritionix API cache cleared")


# Global API instance (initialized lazily)
_nutritionix_api = None

def get_nutritionix_api() -> NutritionixAPI:
    """Get global Nutritionix API instance."""
    global _nutritionix_api
    if _nutritionix_api is None:
        _nutritionix_api = NutritionixAPI()
    return _nutritionix_api