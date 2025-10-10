#!/usr/bin/env python3
"""
Complete 233-Class Nutrition Mapping

Nutritional information for all 233 specific food classes from the ViT classification model.
Based on USDA FoodData Central, food composition databases, and ingredient analysis.
"""

from typing import Dict
# Use absolute import to avoid circular dependencies
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Define NutritionInfo locally to avoid circular import
from dataclasses import dataclass

@dataclass
class NutritionInfo:
    """Nutritional information per 100g of food."""

    calories_per_100g: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    density_g_ml: float

# Comprehensive nutrition mapping for all 233 food classes
COMPLETE_233_NUTRITION: Dict[str, NutritionInfo] = {
    # === BEVERAGES ===
    "alcoholic_beverage": NutritionInfo(235, 0.0, 7.0, 0.0, 0.0, 0.97),
    "bubble_milk_tea": NutritionInfo(120, 2.5, 18.0, 4.0, 0.0, 1.02),
    "fruit_juice": NutritionInfo(46, 0.7, 11.0, 0.2, 0.2, 1.04),
    "instant_cereal_drink": NutritionInfo(75, 3.2, 12.0, 1.5, 1.0, 1.03),
    "kopi_teh_milo": NutritionInfo(45, 1.0, 8.5, 1.2, 0.0, 1.00),
    "chin_chow_drink": NutritionInfo(25, 0.1, 6.0, 0.0, 0.5, 1.00),
    "milk": NutritionInfo(42, 3.4, 5.0, 1.0, 0.0, 1.03),
    "singapore_sling": NutritionInfo(165, 0.0, 14.0, 0.0, 0.0, 0.98),

    # === FRUITS ===
    "apple": NutritionInfo(52, 0.3, 14.0, 0.2, 2.4, 0.83),
    "apricot": NutritionInfo(48, 1.4, 11.0, 0.4, 2.0, 0.85),
    "avocados": NutritionInfo(160, 2.0, 9.0, 15.0, 7.0, 0.96),
    "banana": NutritionInfo(89, 1.1, 23.0, 0.3, 2.6, 0.90),
    "blackberry": NutritionInfo(43, 1.4, 10.0, 0.5, 5.0, 0.92),
    "blueberries": NutritionInfo(57, 0.7, 14.0, 0.3, 2.4, 0.92),
    "cherries": NutritionInfo(63, 1.1, 16.0, 0.2, 2.1, 0.90),
    "cucumber": NutritionInfo(16, 0.7, 4.0, 0.1, 0.5, 0.96),
    "durian": NutritionInfo(147, 1.5, 27.0, 5.3, 3.8, 0.88),
    "grape": NutritionInfo(69, 0.7, 18.0, 0.2, 0.9, 0.80),
    "grapefruit": NutritionInfo(42, 0.8, 11.0, 0.1, 1.6, 0.87),
    "guava": NutritionInfo(68, 2.6, 14.0, 1.0, 5.4, 0.85),
    "honeydew": NutritionInfo(36, 0.5, 9.0, 0.1, 0.8, 0.92),
    "jackfruit": NutritionInfo(95, 1.7, 23.0, 0.6, 1.5, 0.87),
    "kiwi": NutritionInfo(61, 1.1, 15.0, 0.5, 3.0, 0.84),
    "longan": NutritionInfo(60, 1.3, 15.0, 0.1, 1.1, 0.82),
    "lychee": NutritionInfo(66, 0.8, 17.0, 0.4, 1.3, 0.80),
    "mangosteen": NutritionInfo(73, 0.4, 18.0, 0.6, 1.8, 0.84),
    "orange": NutritionInfo(47, 0.9, 12.0, 0.1, 2.4, 0.87),
    "papaya": NutritionInfo(43, 0.5, 11.0, 0.3, 1.7, 0.89),
    "passion_fruit": NutritionInfo(97, 2.2, 23.0, 0.7, 10.0, 0.78),
    "pear": NutritionInfo(57, 0.4, 15.0, 0.1, 3.1, 0.84),
    "persimmon": NutritionInfo(70, 0.6, 19.0, 0.2, 3.6, 0.82),
    "pineapple": NutritionInfo(50, 0.5, 13.0, 0.1, 1.4, 0.89),
    "pitaya": NutritionInfo(60, 1.2, 13.0, 0.4, 3.0, 0.85),
    "pomegranate": NutritionInfo(83, 1.7, 19.0, 1.2, 4.0, 0.78),
    "pomelo": NutritionInfo(38, 0.8, 10.0, 0.0, 1.0, 0.86),
    "rambutan": NutritionInfo(82, 0.7, 21.0, 0.2, 0.9, 0.81),
    "raspberry": NutritionInfo(52, 1.2, 12.0, 0.7, 6.5, 0.92),
    "rock_melon": NutritionInfo(34, 0.8, 8.0, 0.2, 0.9, 0.92),
    "soursop": NutritionInfo(66, 1.0, 17.0, 0.3, 3.3, 0.87),
    "starfruit": NutritionInfo(31, 1.0, 7.0, 0.3, 2.8, 0.91),
    "strawberry": NutritionInfo(32, 0.7, 8.0, 0.3, 2.0, 0.92),
    "watermelon": NutritionInfo(30, 0.6, 8.0, 0.2, 0.4, 0.92),

    # === ASIAN NOODLE DISHES ===
    "laksa": NutritionInfo(285, 12.0, 35.0, 12.0, 2.5, 1.15),
    "ban_mian": NutritionInfo(195, 9.5, 28.0, 5.5, 2.0, 1.12),
    "bee_hoon_goreng": NutritionInfo(245, 8.0, 32.0, 9.5, 1.8, 1.08),
    "bee_hoon_soto": NutritionInfo(185, 12.0, 24.0, 4.5, 1.5, 1.15),
    "bee_hoon": NutritionInfo(165, 5.5, 28.0, 3.0, 1.2, 1.05),
    "dumpling_noodle_soup": NutritionInfo(210, 14.0, 26.0, 6.5, 2.0, 1.18),
    "fish_ball_noodles": NutritionInfo(225, 15.0, 28.0, 6.0, 1.8, 1.16),
    "fried_noodles": NutritionInfo(295, 10.0, 38.0, 12.0, 2.2, 1.10),
    "hor_fun": NutritionInfo(265, 18.0, 32.0, 8.5, 2.0, 1.14),
    "kway_teow": NutritionInfo(275, 15.5, 35.0, 9.0, 1.8, 1.12),
    "lor_mee": NutritionInfo(235, 12.5, 32.0, 7.0, 2.5, 1.15),
    "mee_bandung": NutritionInfo(245, 14.0, 28.0, 8.5, 2.8, 1.08),
    "mee_goreng": NutritionInfo(285, 12.0, 38.0, 11.0, 3.2, 1.06),
    "mee_kuah": NutritionInfo(195, 8.5, 26.0, 5.5, 2.0, 1.12),
    "mee_rebus": NutritionInfo(215, 10.0, 28.0, 7.5, 3.5, 1.14),
    "mee_siam": NutritionInfo(255, 13.0, 32.0, 8.0, 2.5, 1.10),
    "mee_siam_fried": NutritionInfo(295, 12.5, 38.0, 11.5, 2.2, 1.08),
    "mee_soto": NutritionInfo(205, 14.0, 24.0, 6.0, 1.8, 1.15),
    "mee_pok": NutritionInfo(235, 16.0, 28.0, 7.0, 1.5, 1.14),
    "prawn_noodle": NutritionInfo(245, 16.0, 26.0, 8.0, 1.8, 1.16),
    "seafood_noodles_soup": NutritionInfo(225, 18.0, 24.0, 6.5, 1.5, 1.18),
    "wanton_mee_dry": NutritionInfo(285, 15.0, 35.0, 9.5, 2.0, 1.12),
    "beef_noodle_soup": NutritionInfo(265, 20.0, 26.0, 8.5, 2.0, 1.16),
    "hokkien_prawn_mee": NutritionInfo(295, 18.0, 32.0, 10.5, 2.2, 1.14),
    "satay_bee_hoon": NutritionInfo(275, 16.0, 28.0, 11.0, 3.0, 1.12),
    "vegetarian_bee_hoon": NutritionInfo(185, 8.0, 28.0, 5.0, 3.5, 1.05),
    "tom_yum_noodle_soup": NutritionInfo(195, 12.0, 24.0, 5.5, 2.0, 1.08),

    # === RICE DISHES ===
    "claypot_rice": NutritionInfo(245, 18.0, 32.0, 6.5, 1.2, 1.18),
    "duck_rice": NutritionInfo(285, 22.0, 28.0, 9.5, 1.0, 1.20),
    "fried_rice": NutritionInfo(235, 8.5, 32.0, 7.5, 1.5, 1.16),
    "chicken_rice": NutritionInfo(265, 20.0, 30.0, 6.0, 0.8, 1.18),
    "cooked_brown_rice": NutritionInfo(112, 2.3, 23.0, 0.9, 1.8, 1.25),
    "cooked_white_rice": NutritionInfo(130, 2.7, 28.0, 0.3, 0.4, 1.25),
    "nasi_lemak": NutritionInfo(295, 8.5, 38.0, 12.0, 2.5, 1.12),
    "nasi_ambeng": NutritionInfo(315, 18.0, 35.0, 11.5, 2.8, 1.15),
    "nasi_pattaya": NutritionInfo(285, 16.0, 32.0, 9.0, 1.5, 1.16),
    "yam_rice": NutritionInfo(195, 12.0, 28.0, 5.5, 2.2, 1.14),
    "thunder_tea_rice": NutritionInfo(165, 8.0, 25.0, 4.0, 3.5, 1.08),
    "tumpeng": NutritionInfo(185, 6.5, 28.0, 5.5, 2.0, 1.12),

    # === JAPANESE DISHES ===
    "don_chicken_teriyaki": NutritionInfo(295, 22.0, 32.0, 8.0, 1.2, 1.18),
    "don_unagi": NutritionInfo(325, 20.0, 35.0, 12.0, 1.0, 1.16),
    "katsudon": NutritionInfo(345, 18.0, 38.0, 14.0, 2.0, 1.14),
    "miso_ramen_with_fishcake": NutritionInfo(265, 14.0, 32.0, 8.5, 2.5, 1.12),
    "udon": NutritionInfo(175, 6.0, 28.0, 4.0, 1.8, 1.08),
    "chawanmushi": NutritionInfo(125, 12.0, 5.0, 6.5, 0.5, 1.05),
    "miso_soup": NutritionInfo(45, 3.5, 5.0, 1.2, 0.8, 1.02),
    "sushi": NutritionInfo(185, 15.0, 22.0, 4.5, 0.8, 1.15),

    # === KOREAN DISHES ===
    "bibimbap": NutritionInfo(265, 16.0, 32.0, 8.0, 3.5, 1.12),
    "rice_korean_bulgogi_beef": NutritionInfo(295, 22.0, 28.0, 9.5, 1.5, 1.18),
    "rice_with_dakgalbi": NutritionInfo(275, 19.0, 30.0, 8.0, 2.0, 1.16),

    # === INDIAN/MALAY DISHES ===
    "indian_pancake": NutritionInfo(245, 6.5, 35.0, 8.5, 2.0, 0.85),
    "indian_prata": NutritionInfo(265, 7.0, 38.0, 9.5, 1.8, 0.82),
    "lontong_with_sayur_lodeh": NutritionInfo(185, 8.0, 25.0, 6.5, 3.2, 1.08),
    "lontong": NutritionInfo(145, 4.5, 22.0, 4.0, 1.8, 1.05),
    "soto_ayam": NutritionInfo(165, 16.0, 12.0, 5.5, 1.5, 1.12),
    "assam_pedas": NutritionInfo(135, 18.0, 8.0, 4.0, 2.5, 1.02),
    "ayam_penyet": NutritionInfo(285, 24.0, 18.0, 12.0, 1.2, 1.15),
    "gulai_daun_ubi": NutritionInfo(95, 3.5, 12.0, 4.0, 4.5, 0.98),
    "murtabak": NutritionInfo(315, 14.0, 28.0, 16.5, 2.2, 0.95),
    "vadai": NutritionInfo(285, 12.0, 22.0, 18.0, 5.5, 0.85),

    # === BREAD & BAKERY ===
    "white_bread": NutritionInfo(265, 9.0, 49.0, 3.2, 2.7, 0.28),
    "whole_grain_bread": NutritionInfo(247, 13.0, 41.0, 4.2, 7.0, 0.41),
    "wholegrain_wrap": NutritionInfo(225, 11.5, 38.0, 4.8, 6.5, 0.45),
    "bagel_croissant": NutritionInfo(365, 8.5, 45.0, 16.0, 2.5, 0.65),
    "bagel_and_croissant": NutritionInfo(365, 8.5, 45.0, 16.0, 2.5, 0.65),
    "biscuit": NutritionInfo(485, 6.8, 58.0, 24.0, 2.2, 0.55),
    "cake": NutritionInfo(365, 6.5, 52.0, 14.5, 1.8, 0.75),
    "cake_rolls": NutritionInfo(285, 5.8, 38.0, 12.0, 1.5, 0.80),
    "pancake": NutritionInfo(225, 6.0, 32.0, 8.0, 1.8, 0.85),
    "waffle": NutritionInfo(265, 6.8, 35.0, 11.0, 2.0, 0.80),

    # === BREAKFAST ===
    "breakfast_cereal": NutritionInfo(85, 3.5, 15.0, 1.8, 2.5, 0.82),
    "muesli": NutritionInfo(385, 12.0, 58.0, 12.5, 8.5, 0.75),
    "whole_oats": NutritionInfo(68, 2.5, 12.0, 1.4, 1.7, 0.84),
    "whole_wheat": NutritionInfo(340, 13.0, 72.0, 2.5, 12.0, 0.59),
    "wholegrain_muffin": NutritionInfo(285, 8.5, 45.0, 8.5, 5.5, 0.68),
    "soft_boiled_eggs": NutritionInfo(155, 13.0, 1.1, 11.0, 0.0, 1.03),
    "kaya_toast": NutritionInfo(245, 6.0, 35.0, 8.5, 2.0, 0.55),

    # === GRAINS & CEREALS ===
    "barley": NutritionInfo(123, 2.3, 28.0, 0.4, 3.8, 1.22),
    "buckwheat": NutritionInfo(343, 13.0, 72.0, 3.4, 10.0, 1.20),
    "corn": NutritionInfo(86, 3.3, 19.0, 1.4, 2.7, 0.88),
    "porridge": NutritionInfo(68, 2.5, 12.0, 1.4, 1.7, 0.84),

    # === SOUPS ===
    "chicken_soup": NutritionInfo(85, 12.0, 6.0, 2.5, 1.2, 1.08),
    "cream_soup": NutritionInfo(135, 4.5, 12.0, 8.0, 1.8, 1.02),
    "mushroom_soup": NutritionInfo(95, 3.8, 8.5, 5.5, 2.2, 0.98),
    "vegetable_soup": NutritionInfo(45, 2.2, 8.0, 1.0, 2.5, 0.95),
    "pig_organ_soup": NutritionInfo(125, 16.0, 5.5, 4.5, 1.0, 1.08),
    "sliced_fish_soup": NutritionInfo(95, 15.0, 3.5, 2.5, 0.8, 1.06),

    # === MEAT DISHES ===
    "burger": NutritionInfo(295, 16.0, 28.0, 13.5, 2.5, 0.95),
    "fried_chicken": NutritionInfo(320, 19.0, 15.0, 20.0, 1.2, 0.98),
    "chicken_chop": NutritionInfo(245, 25.0, 5.0, 12.0, 1.8, 1.05),
    "chicken_masala": NutritionInfo(285, 18.0, 12.0, 18.5, 2.5, 1.02),
    "chicken_pie": NutritionInfo(285, 14.0, 25.0, 16.0, 2.8, 0.88),
    "chicken_wing": NutritionInfo(295, 18.5, 3.0, 22.0, 0.5, 1.02),
    "roasted_chicken": NutritionInfo(265, 24.0, 2.0, 16.5, 0.8, 1.04),
    "tandoori_chicken": NutritionInfo(245, 20.0, 8.0, 14.0, 1.2, 1.01),
    "har_cheong_gai": NutritionInfo(285, 18.0, 18.0, 16.0, 1.5, 0.95),

    # === SEAFOOD ===
    "fish_and_chips": NutritionInfo(365, 16.0, 32.0, 19.5, 2.8, 0.95),
    "fried_fish": NutritionInfo(285, 18.0, 12.0, 18.5, 1.0, 1.02),
    "fried_prawn": NutritionInfo(295, 16.5, 14.0, 19.0, 1.2, 1.05),
    "cereal_prawns": NutritionInfo(265, 15.0, 18.0, 15.0, 2.5, 1.02),
    "drunken_prawn": NutritionInfo(185, 19.0, 4.0, 8.5, 0.5, 1.06),
    "black_pepper_crab": NutritionInfo(225, 16.5, 8.0, 13.5, 2.0, 1.04),
    "chilli_crab": NutritionInfo(265, 15.0, 12.0, 16.5, 2.5, 1.02),
    "fish_head_curry": NutritionInfo(195, 16.0, 8.5, 11.0, 2.8, 1.00),
    "salmon_grilled": NutritionInfo(225, 22.0, 2.0, 13.0, 0.5, 1.04),
    "salmon___grilled": NutritionInfo(225, 22.0, 2.0, 13.0, 0.5, 1.04),
    "steamed_grouper": NutritionInfo(165, 20.0, 3.0, 7.5, 0.8, 1.06),
    "sambal_stingray": NutritionInfo(185, 18.0, 6.0, 9.5, 1.5, 1.04),

    # === PORK & BEEF ===
    "bak_chor_mee": NutritionInfo(285, 16.0, 32.0, 10.5, 2.2, 1.08),
    "bak_kut_teh": NutritionInfo(195, 18.0, 8.0, 9.5, 1.5, 1.05),
    "bak_kwa": NutritionInfo(365, 22.0, 25.0, 19.0, 1.0, 1.02),
    "char_siew": NutritionInfo(295, 20.0, 18.0, 16.5, 0.8, 1.01),
    "char_siew_pau": NutritionInfo(265, 14.0, 28.0, 11.0, 2.0, 0.88),
    "kebab_beef": NutritionInfo(265, 18.0, 22.0, 12.0, 2.5, 0.90),
    "kebab_chicken": NutritionInfo(245, 19.0, 20.0, 10.5, 2.2, 0.92),
    "kebab___beef": NutritionInfo(265, 18.0, 22.0, 12.0, 2.5, 0.90),
    "kebab___chicken": NutritionInfo(245, 19.0, 20.0, 10.5, 2.2, 0.92),
    "lamb_chops": NutritionInfo(285, 22.0, 2.0, 20.0, 0.5, 1.02),
    "sirloin_steak": NutritionInfo(265, 24.0, 1.0, 17.0, 0.2, 1.04),
    "satay": NutritionInfo(295, 18.0, 12.0, 19.5, 3.2, 0.95),

    # === VEGETABLES ===
    "green_leafy_vegetables": NutritionInfo(23, 2.9, 4.0, 0.4, 2.2, 0.91),
    "mixed_vegetables": NutritionInfo(32, 2.2, 6.5, 0.3, 2.8, 0.89),
    "sambal_kangkung": NutritionInfo(45, 3.2, 6.0, 1.5, 2.8, 0.88),
    "salad": NutritionInfo(18, 1.8, 3.5, 0.2, 2.2, 0.94),
    "baked_beans": NutritionInfo(125, 8.0, 22.0, 1.2, 6.5, 1.12),

    # === TOFU & SOY ===
    "tauhu_goreng": NutritionInfo(185, 12.0, 8.5, 12.0, 2.0, 0.95),
    "hotplate_tofu": NutritionInfo(145, 10.5, 6.0, 9.0, 2.5, 0.98),
    "yong_tau_foo": NutritionInfo(125, 12.0, 8.0, 5.5, 1.8, 1.02),

    # === PASTA & WESTERN ===
    "macaroni": NutritionInfo(185, 8.5, 24.0, 6.0, 1.5, 1.08),
    "pasta_fettuccine": NutritionInfo(235, 9.0, 28.0, 10.0, 1.8, 1.05),
    "pasta_red_sauce": NutritionInfo(195, 7.5, 28.0, 6.5, 2.2, 1.02),
    "pasta___fettuccine": NutritionInfo(235, 9.0, 28.0, 10.0, 1.8, 1.05),
    "pasta___red_sauce": NutritionInfo(195, 7.5, 28.0, 6.5, 2.2, 1.02),
    "wholegrain_pasta": NutritionInfo(145, 6.0, 25.0, 2.5, 3.5, 1.20),
    "lasagna": NutritionInfo(285, 16.0, 25.0, 14.0, 2.8, 1.02),
    "pizza": NutritionInfo(266, 11.0, 33.0, 10.0, 2.3, 0.85),
    "sandwich": NutritionInfo(250, 12.0, 30.0, 10.0, 3.0, 0.70),
    "french_fries": NutritionInfo(365, 4.0, 48.0, 17.0, 4.0, 0.85),
    "cheese_fries": NutritionInfo(385, 12.0, 42.0, 21.0, 3.5, 0.88),
    "mixed_grills": NutritionInfo(295, 26.0, 5.0, 18.5, 2.0, 1.02),

    # === SNACKS ===
    "chocolate": NutritionInfo(535, 7.6, 60.0, 31.0, 7.0, 1.20),
    "nuts": NutritionInfo(607, 20.0, 16.0, 54.0, 7.0, 0.65),
    "snacks_and_chips": NutritionInfo(465, 6.5, 52.0, 25.0, 4.5, 0.68),
    "sweets": NutritionInfo(385, 1.5, 85.0, 4.5, 0.5, 1.35),
    "popcorn": NutritionInfo(385, 12.0, 78.0, 4.5, 15.0, 0.52),
    "preserved_fruit_snacks": NutritionInfo(265, 2.5, 62.0, 1.8, 4.5, 1.15),
    "seaweed_snack": NutritionInfo(165, 12.0, 8.5, 9.0, 4.0, 0.85),

    # === DESSERTS ===
    "ice_cream_chocolate": NutritionInfo(216, 3.8, 23.0, 11.0, 1.2, 1.08),
    "ice_cream_vanilla": NutritionInfo(207, 3.5, 24.0, 11.0, 0.7, 1.08),
    "ice_cream___chocolate": NutritionInfo(216, 3.8, 23.0, 11.0, 1.2, 1.08),
    "ice_cream___vanilla": NutritionInfo(207, 3.5, 24.0, 11.0, 0.7, 1.08),
    "ice_kacang": NutritionInfo(125, 2.5, 28.0, 1.5, 1.8, 0.95),
    "tiramisu": NutritionInfo(285, 6.5, 32.0, 14.5, 1.5, 0.98),
    "parfait": NutritionInfo(135, 8.5, 18.0, 3.5, 2.2, 1.02),
    "mango_pudding": NutritionInfo(125, 2.8, 25.0, 2.5, 1.2, 0.92),

    # === TRADITIONAL SWEETS ===
    "cny_love_letter": NutritionInfo(465, 6.5, 58.0, 22.0, 2.5, 0.75),
    "kueh_salat": NutritionInfo(195, 3.5, 32.0, 6.5, 1.8, 1.08),
    "peanut_pancake": NutritionInfo(385, 12.0, 45.0, 18.0, 4.5, 0.85),
    "pung_kueh": NutritionInfo(185, 4.0, 32.0, 4.5, 1.5, 0.88),
    "kueh_lapis_rainbow": NutritionInfo(265, 4.5, 38.0, 10.5, 1.8, 0.85),
    "kueh_lapis___rainbow": NutritionInfo(265, 4.5, 38.0, 10.5, 1.8, 0.85),
    "kueh_lapis_baked": NutritionInfo(285, 6.5, 42.0, 10.0, 1.5, 0.82),
    "kuih_bahulu": NutritionInfo(245, 7.5, 38.0, 8.0, 1.2, 0.75),
    "pineapple_tarts": NutritionInfo(365, 5.5, 52.0, 15.0, 2.8, 0.82),
    "tau_suan": NutritionInfo(125, 6.5, 22.0, 1.5, 4.5, 1.05),
    "tutu_kueh": NutritionInfo(165, 3.5, 28.0, 4.5, 1.2, 1.02),
    "egg_tart": NutritionInfo(285, 8.5, 32.0, 14.0, 1.0, 0.92),
    "cheng_tng": NutritionInfo(85, 3.0, 18.0, 0.8, 2.5, 1.02),

    # === DUMPLINGS & WRAPPED FOODS ===
    "dumpling": NutritionInfo(235, 12.0, 25.0, 9.5, 2.2, 0.95),
    "siew_mai": NutritionInfo(245, 14.0, 18.0, 12.0, 1.8, 0.98),
    "rice_dumpling": NutritionInfo(215, 11.0, 28.0, 7.0, 1.5, 1.08),
    "xiao_long_bao": NutritionInfo(265, 12.5, 22.0, 14.0, 1.8, 0.92),
    "spring_rolls": NutritionInfo(185, 8.5, 22.0, 7.5, 3.2, 0.88),
    "popiah": NutritionInfo(165, 9.0, 18.0, 6.5, 3.5, 0.85),
    "curry_puff": NutritionInfo(265, 8.0, 28.0, 13.5, 2.8, 0.82),
    "chwee_kueh": NutritionInfo(185, 8.5, 24.0, 6.0, 1.2, 1.05),

    # === TRADITIONAL DISHES ===
    "chap_chye_nonya": NutritionInfo(85, 5.5, 12.0, 2.5, 3.8, 0.92),
    "chinese_fritters": NutritionInfo(385, 8.0, 42.0, 20.0, 2.5, 0.78),
    "chai_tow_kuay": NutritionInfo(195, 8.5, 22.0, 8.5, 1.0, 1.02),
    "goreng_pisang": NutritionInfo(245, 3.5, 38.0, 8.5, 2.8, 0.82),
    "begedil": NutritionInfo(165, 5.5, 18.0, 7.5, 2.2, 0.95),
    "ngoh_hiang": NutritionInfo(245, 14.0, 12.0, 16.0, 2.5, 0.92),
    "otak": NutritionInfo(185, 14.0, 6.0, 11.5, 1.5, 0.95),
    "oyster_omelette": NutritionInfo(195, 14.0, 12.0, 11.0, 1.2, 0.98),
    "rojak": NutritionInfo(125, 6.5, 18.0, 4.0, 4.5, 0.85),
    "roti_john": NutritionInfo(245, 12.0, 25.0, 11.5, 2.2, 0.85),
    "sambal": NutritionInfo(65, 2.5, 8.5, 2.5, 3.0, 0.85),
    "kway_chap": NutritionInfo(215, 14.0, 22.0, 7.5, 1.8, 1.08),
    "meatball": NutritionInfo(225, 18.0, 8.0, 13.0, 1.2, 1.05),
    "bakso": NutritionInfo(185, 16.0, 12.0, 8.5, 1.5, 1.08),
    "sausage_rolls": NutritionInfo(285, 12.0, 22.0, 17.0, 2.0, 0.85),
    "steamed_buns": NutritionInfo(195, 8.5, 28.0, 5.5, 2.2, 0.82),
    "yuseng": NutritionInfo(165, 14.0, 12.0, 7.5, 2.8, 0.92),

    # === INTERNATIONAL ===
    "paella_seafood": NutritionInfo(235, 16.0, 25.0, 8.5, 1.8, 1.12),
    "tacos_and_nachos": NutritionInfo(295, 14.0, 32.0, 12.5, 3.5, 0.85),
    "tortilla_plain": NutritionInfo(285, 7.5, 45.0, 8.0, 3.2, 0.78),
    "tortilla___plain": NutritionInfo(285, 7.5, 45.0, 8.0, 3.2, 0.78),

    # === SPECIAL RICE DISHES ===
    "rice_chicken_katsu_with_japanese_curry": NutritionInfo(315, 18.0, 35.0, 11.5, 2.5, 1.15),
    "pilaf_pea": NutritionInfo(165, 5.5, 28.0, 3.5, 2.8, 1.18),
    "yogurt": NutritionInfo(59, 10.0, 3.6, 0.4, 0.0, 1.04),

    # === CANNED/PROCESSED ===
    "canned_fruit_salad": NutritionInfo(85, 0.8, 20.0, 0.3, 1.8, 0.92),
}