#!/usr/bin/env python3
"""
Complete 244-Class Food Mapping

Complete ingredient breakdown for all 244 food classes from the ViT classification model.
This will replace the existing mapping in food_ingredient_mapping.py
"""

# Complete mapping for all 244 food classes
COMPLETE_244_MAPPINGS = {
    # === BEVERAGES ===
    "alcoholic_beverage": [("water", 0.85), ("sugar_white", 0.10), ("herbs_dried", 0.05)],
    "bubble_milk_tea": [("milk_whole", 0.70), ("sugar_white", 0.20), ("water", 0.10)],
    "fruit_juice": [("oranges", 0.88), ("water", 0.12)],
    "instant_cereal_drink": [("milk_whole", 0.80), ("oats_cooked", 0.15), ("sugar_white", 0.05)],
    "kopi_teh_milo": [("water", 0.85), ("milk_whole", 0.10), ("sugar_white", 0.05)],
    "chin_chow_drink": [("water", 0.90), ("sugar_white", 0.08), ("herbs_dried", 0.02)],
    "milk": [("milk_whole", 1.0)],
    "singapore_sling": [("water", 0.70), ("oranges", 0.25), ("sugar_white", 0.05)],

    # === FRUITS (SINGLE INGREDIENTS) ===
    "apple": [("apples", 1.0)],
    "apricot": [("apples", 1.0)],
    "avocados": [("apples", 0.70), ("oil_olive", 0.30)],
    "banana": [("bananas", 1.0)],
    "blackberry": [("strawberries", 1.0)],
    "blueberries": [("strawberries", 1.0)],
    "cherries": [("strawberries", 1.0)],
    "cucumber": [("cucumber", 1.0)],
    "durian": [("bananas", 0.60), ("cream_heavy", 0.40)],
    "grape": [("grapes", 1.0)],
    "grapefruit": [("oranges", 1.0)],
    "guava": [("apples", 1.0)],
    "honeydew": [("cucumber", 0.90), ("sugar_white", 0.10)],
    "jackfruit": [("pineapple", 0.80), ("sugar_white", 0.20)],
    "kiwi": [("apples", 1.0)],
    "longan": [("grapes", 1.0)],
    "lychee": [("grapes", 1.0)],
    "mangosteen": [("apples", 1.0)],
    "orange": [("oranges", 1.0)],
    "papaya": [("pineapple", 1.0)],
    "passion_fruit": [("oranges", 0.60), ("sugar_white", 0.40)],
    "pear": [("apples", 1.0)],
    "persimmon": [("apples", 1.0)],
    "pineapple": [("pineapple", 1.0)],
    "pitaya": [("apples", 1.0)],
    "pomegranate": [("grapes", 0.80), ("sugar_white", 0.20)],
    "pomelo": [("oranges", 1.0)],
    "rambutan": [("grapes", 1.0)],
    "raspberry": [("strawberries", 1.0)],
    "rock_melon": [("cucumber", 0.90), ("sugar_white", 0.10)],
    "soursop": [("apples", 0.80), ("cream_heavy", 0.20)],
    "starfruit": [("apples", 1.0)],
    "strawberry": [("strawberries", 1.0)],
    "watermelon": [("cucumber", 0.92), ("sugar_white", 0.08)],

    # === ASIAN NOODLE DISHES ===
    "laksa": [("rice_noodles", 0.38), ("coconut_milk", 0.32), ("shrimp", 0.15), ("vegetables", 0.10), ("herbs_fresh", 0.05)],
    "ban_mian": [("wheat_noodles", 0.45), ("broth_chicken", 0.30), ("vegetables", 0.15), ("pork_lean", 0.10)],
    "bee_hoon_goreng": [("rice_noodles", 0.50), ("vegetables", 0.25), ("shrimp", 0.15), ("oil_vegetable", 0.10)],
    "bee_hoon_soto": [("rice_noodles", 0.40), ("broth_chicken", 0.35), ("chicken_breast", 0.15), ("vegetables", 0.10)],
    "bee_hoon": [("rice_noodles", 0.60), ("broth_vegetable", 0.30), ("vegetables", 0.10)],
    "dumpling_noodle_soup": [("wheat_noodles", 0.35), ("broth_chicken", 0.35), ("pork_lean", 0.20), ("flour_wheat", 0.10)],
    "fish_ball_noodles": [("rice_noodles", 0.45), ("broth_chicken", 0.30), ("fish_white", 0.20), ("vegetables", 0.05)],
    "fried_noodles": [("wheat_noodles", 0.55), ("vegetables", 0.25), ("oil_vegetable", 0.12), ("soy_sauce", 0.08)],
    "hor_fun": [("rice_noodles", 0.50), ("beef_lean", 0.25), ("vegetables", 0.20), ("oil_vegetable", 0.05)],
    "kway_teow": [("rice_noodles", 0.50), ("shrimp", 0.20), ("vegetables", 0.20), ("oil_vegetable", 0.10)],
    "lor_mee": [("wheat_noodles", 0.45), ("broth_chicken", 0.30), ("pork_lean", 0.15), ("cornstarch", 0.10)],
    "mee_bandung": [("wheat_noodles", 0.40), ("tomato_sauce", 0.25), ("shrimp", 0.20), ("vegetables", 0.15)],
    "mee_goreng": [("wheat_noodles", 0.50), ("vegetables", 0.25), ("oil_vegetable", 0.15), ("tofu_firm", 0.10)],
    "mee_kuah": [("wheat_noodles", 0.45), ("broth_chicken", 0.35), ("vegetables", 0.15), ("herbs_fresh", 0.05)],
    "mee_rebus": [("wheat_noodles", 0.40), ("broth_chicken", 0.30), ("potatoes", 0.20), ("peanuts", 0.10)],
    "mee_siam": [("rice_noodles", 0.45), ("tomato_sauce", 0.25), ("shrimp", 0.20), ("vegetables", 0.10)],
    "mee_siam_fried": [("rice_noodles", 0.50), ("vegetables", 0.25), ("oil_vegetable", 0.15), ("shrimp", 0.10)],
    "mee_soto": [("wheat_noodles", 0.40), ("broth_chicken", 0.35), ("chicken_breast", 0.15), ("vegetables", 0.10)],
    "mee_pok": [("wheat_noodles", 0.50), ("fish_white", 0.25), ("vegetables", 0.20), ("oil_vegetable", 0.05)],
    "prawn_noodle": [("rice_noodles", 0.45), ("broth_chicken", 0.30), ("shrimp", 0.20), ("vegetables", 0.05)],
    "seafood_noodles_soup": [("rice_noodles", 0.40), ("broth_chicken", 0.30), ("shrimp", 0.15), ("fish_white", 0.15)],
    "wanton_mee_dry": [("wheat_noodles", 0.50), ("pork_lean", 0.25), ("vegetables", 0.20), ("oil_vegetable", 0.05)],
    "beef_noodle_soup": [("wheat_noodles", 0.40), ("broth_beef", 0.35), ("beef_lean", 0.20), ("vegetables", 0.05)],
    "hokkien_prawn_mee": [("wheat_noodles", 0.40), ("rice_noodles", 0.20), ("shrimp", 0.25), ("broth_chicken", 0.15)],
    "satay_bee_hoon": [("rice_noodles", 0.45), ("chicken_breast", 0.25), ("peanuts", 0.20), ("vegetables", 0.10)],
    "vegetarian_bee_hoon": [("rice_noodles", 0.60), ("vegetables", 0.30), ("tofu_firm", 0.10)],
    "tom_yum_noodle_soup": [("rice_noodles", 0.40), ("broth_vegetable", 0.35), ("shrimp", 0.15), ("herbs_fresh", 0.10)],

    # === RICE DISHES ===
    "claypot_rice": [("rice_cooked", 0.60), ("chicken_thigh", 0.25), ("vegetables", 0.10), ("oil_vegetable", 0.05)],
    "duck_rice": [("rice_cooked", 0.60), ("chicken_thigh", 0.30), ("vegetables", 0.08), ("oil_vegetable", 0.02)],
    "fried_rice": [("rice_cooked", 0.55), ("egg_whole", 0.20), ("vegetables", 0.15), ("oil_vegetable", 0.10)],
    "chicken_rice": [("rice_cooked", 0.60), ("chicken_breast", 0.30), ("broth_chicken", 0.08), ("oil_vegetable", 0.02)],
    "cooked_brown_rice": [("rice_cooked", 1.0)],
    "cooked_white_rice": [("rice_cooked", 1.0)],
    "nasi_lemak": [("rice_cooked", 0.50), ("coconut_milk", 0.30), ("peanuts", 0.10), ("vegetables", 0.10)],
    "nasi_ambeng": [("rice_cooked", 0.45), ("beef_regular", 0.25), ("vegetables", 0.20), ("coconut_milk", 0.10)],
    "nasi_pattaya": [("rice_cooked", 0.50), ("egg_whole", 0.30), ("chicken_breast", 0.15), ("vegetables", 0.05)],
    "yam_rice": [("rice_cooked", 0.60), ("sweet_potatoes", 0.25), ("chicken_breast", 0.10), ("oil_vegetable", 0.05)],
    "thunder_tea_rice": [("rice_cooked", 0.50), ("vegetables", 0.35), ("peanuts", 0.10), ("herbs_fresh", 0.05)],
    "tumpeng": [("rice_cooked", 0.60), ("coconut_milk", 0.25), ("vegetables", 0.10), ("herbs_fresh", 0.05)],

    # === JAPANESE DISHES ===
    "don_chicken_teriyaki": [("rice_cooked", 0.50), ("chicken_thigh", 0.35), ("soy_sauce", 0.10), ("vegetables", 0.05)],
    "don_unagi": [("rice_cooked", 0.50), ("fish_oily", 0.35), ("soy_sauce", 0.10), ("vegetables", 0.05)],
    "katsudon": [("rice_cooked", 0.45), ("pork_lean", 0.30), ("egg_whole", 0.15), ("bread_crumbs", 0.10)],
    "miso_ramen_with_fishcake": [("wheat_noodles", 0.40), ("broth_vegetable", 0.35), ("fish_white", 0.15), ("vegetables", 0.10)],
    "udon": [("wheat_noodles", 0.60), ("broth_vegetable", 0.30), ("vegetables", 0.08), ("herbs_fresh", 0.02)],
    "chawanmushi": [("egg_whole", 0.60), ("broth_chicken", 0.30), ("shrimp", 0.08), ("vegetables", 0.02)],
    "miso_soup": [("broth_vegetable", 0.80), ("tofu_soft", 0.15), ("vegetables", 0.05)],
    "sushi": [("rice_cooked", 0.50), ("fish_white", 0.35), ("vegetables", 0.12), ("oil_sesame", 0.03)],

    # === KOREAN DISHES ===
    "bibimbap": [("rice_cooked", 0.40), ("vegetables", 0.30), ("beef_lean", 0.20), ("egg_whole", 0.10)],
    "rice_korean_bulgogi_beef": [("rice_cooked", 0.50), ("beef_lean", 0.35), ("vegetables", 0.12), ("oil_vegetable", 0.03)],
    "rice_with_dakgalbi": [("rice_cooked", 0.50), ("chicken_thigh", 0.30), ("vegetables", 0.15), ("spices_ground", 0.05)],

    # === INDIAN/MALAY DISHES ===
    "indian_pancake": [("flour_wheat", 0.60), ("milk_whole", 0.25), ("oil_vegetable", 0.10), ("sugar_white", 0.05)],
    "indian_prata": [("flour_wheat", 0.70), ("oil_vegetable", 0.20), ("milk_whole", 0.10)],
    "lontong_with_sayur_lodeh": [("rice_cooked", 0.45), ("coconut_milk", 0.30), ("vegetables", 0.20), ("tofu_firm", 0.05)],
    "lontong": [("rice_cooked", 0.80), ("coconut_milk", 0.15), ("herbs_fresh", 0.05)],
    "soto_ayam": [("broth_chicken", 0.50), ("chicken_breast", 0.25), ("rice_cooked", 0.20), ("vegetables", 0.05)],
    "assam_pedas": [("fish_white", 0.40), ("tomatoes", 0.30), ("broth_vegetable", 0.25), ("spices_ground", 0.05)],
    "ayam_penyet": [("chicken_thigh", 0.60), ("rice_cooked", 0.30), ("vegetables", 0.08), ("spices_ground", 0.02)],
    "gulai_daun_ubi": [("vegetables", 0.60), ("coconut_milk", 0.30), ("spices_ground", 0.10)],
    "murtabak": [("flour_wheat", 0.50), ("egg_whole", 0.25), ("onions", 0.15), ("oil_vegetable", 0.10)],
    "vadai": [("lentils_cooked", 0.70), ("oil_vegetable", 0.20), ("spices_ground", 0.10)],

    # === BREAD & BAKERY ===
    "white_bread": [("bread_white", 1.0)],
    "whole_grain_bread": [("bread_whole_wheat", 1.0)],
    "wholegrain_wrap": [("bread_whole_wheat", 0.80), ("vegetables", 0.15), ("oil_olive", 0.05)],
    "bagel_croissant": [("flour_wheat", 0.60), ("butter", 0.25), ("milk_whole", 0.10), ("sugar_white", 0.05)],
    "bagel_and_croissant": [("flour_wheat", 0.60), ("butter", 0.25), ("milk_whole", 0.10), ("sugar_white", 0.05)],
    "biscuit": [("flour_wheat", 0.50), ("butter", 0.30), ("sugar_white", 0.15), ("milk_whole", 0.05)],
    "cake": [("flour_wheat", 0.35), ("sugar_white", 0.25), ("egg_whole", 0.20), ("butter", 0.20)],
    "cake_rolls": [("flour_wheat", 0.40), ("sugar_white", 0.25), ("cream_heavy", 0.20), ("egg_whole", 0.15)],
    "pancake": [("flour_wheat", 0.45), ("milk_whole", 0.30), ("egg_whole", 0.20), ("butter", 0.05)],
    "waffle": [("flour_wheat", 0.45), ("milk_whole", 0.25), ("egg_whole", 0.20), ("butter", 0.10)],

    # === BREAKFAST ===
    "breakfast_cereal": [("oats_cooked", 0.80), ("milk_whole", 0.15), ("sugar_white", 0.05)],
    "muesli": [("oats_cooked", 0.70), ("nuts_mixed", 0.20), ("strawberries", 0.10)],
    "whole_oats": [("oats_cooked", 1.0)],
    "whole_wheat": [("flour_wheat", 1.0)],
    "wholegrain_muffin": [("bread_whole_wheat", 0.70), ("sugar_white", 0.20), ("oil_vegetable", 0.10)],
    "soft_boiled_eggs": [("egg_whole", 1.0)],
    "kaya_toast": [("bread_white", 0.70), ("sugar_white", 0.20), ("coconut_milk", 0.08), ("egg_whole", 0.02)],

    # === GRAINS & CEREALS ===
    "barley": [("barley_cooked", 1.0)],
    "buckwheat": [("barley_cooked", 1.0)],  # Similar grain density
    "corn": [("vegetables", 1.0)],  # Corn as vegetable
    "porridge": [("oats_cooked", 0.80), ("milk_whole", 0.15), ("water", 0.05)],

    # === SOUPS ===
    "chicken_soup": [("broth_chicken", 0.60), ("chicken_breast", 0.20), ("vegetables", 0.15), ("rice_noodles", 0.05)],
    "cream_soup": [("cream_heavy", 0.50), ("vegetables", 0.30), ("broth_chicken", 0.15), ("flour_wheat", 0.05)],
    "mushroom_soup": [("mushrooms", 0.40), ("cream_heavy", 0.30), ("broth_vegetable", 0.25), ("flour_wheat", 0.05)],
    "vegetable_soup": [("vegetables", 0.60), ("broth_vegetable", 0.35), ("herbs_fresh", 0.05)],
    "pig_organ_soup": [("pork_lean", 0.40), ("broth_chicken", 0.45), ("vegetables", 0.10), ("herbs_fresh", 0.05)],
    "sliced_fish_soup": [("fish_white", 0.40), ("broth_chicken", 0.45), ("vegetables", 0.12), ("herbs_fresh", 0.03)],

    # === MEAT DISHES ===
    "burger": [("bread_white", 0.35), ("beef_regular", 0.30), ("vegetables", 0.20), ("cheese_soft", 0.15)],
    "fried_chicken": [("chicken_thigh", 0.70), ("flour_wheat", 0.20), ("oil_vegetable", 0.10)],
    "chicken_chop": [("chicken_breast", 0.80), ("vegetables", 0.15), ("oil_vegetable", 0.05)],
    "chicken_masala": [("chicken_thigh", 0.50), ("tomato_sauce", 0.25), ("coconut_milk", 0.20), ("spices_ground", 0.05)],
    "chicken_pie": [("chicken_breast", 0.40), ("flour_wheat", 0.35), ("vegetables", 0.20), ("butter", 0.05)],
    "chicken_wing": [("chicken_thigh", 0.85), ("oil_vegetable", 0.10), ("spices_ground", 0.05)],
    "roasted_chicken": [("chicken_breast", 0.85), ("oil_vegetable", 0.10), ("herbs_fresh", 0.05)],
    "tandoori_chicken": [("chicken_thigh", 0.70), ("yogurt_plain", 0.20), ("spices_ground", 0.10)],
    "har_cheong_gai": [("chicken_thigh", 0.70), ("flour_wheat", 0.20), ("oil_vegetable", 0.10)],

    # === SEAFOOD ===
    "fish_and_chips": [("fish_white", 0.40), ("potatoes", 0.35), ("flour_wheat", 0.15), ("oil_vegetable", 0.10)],
    "fried_fish": [("fish_white", 0.70), ("flour_wheat", 0.20), ("oil_vegetable", 0.10)],
    "fried_prawn": [("shrimp", 0.70), ("flour_wheat", 0.20), ("oil_vegetable", 0.10)],
    "cereal_prawns": [("shrimp", 0.60), ("oats_cooked", 0.25), ("oil_vegetable", 0.15)],
    "drunken_prawn": [("shrimp", 0.80), ("wine_cooking", 0.15), ("herbs_fresh", 0.05)],
    "black_pepper_crab": [("crab", 0.70), ("vegetables", 0.20), ("oil_vegetable", 0.07), ("spices_ground", 0.03)],
    "chilli_crab": [("crab", 0.65), ("tomato_sauce", 0.25), ("egg_whole", 0.07), ("spices_ground", 0.03)],
    "fish_head_curry": [("fish_white", 0.45), ("coconut_milk", 0.30), ("vegetables", 0.20), ("spices_ground", 0.05)],
    "salmon_grilled": [("fish_oily", 0.85), ("oil_olive", 0.10), ("herbs_fresh", 0.05)],
    "salmon___grilled": [("fish_oily", 0.85), ("oil_olive", 0.10), ("herbs_fresh", 0.05)],
    "steamed_grouper": [("fish_white", 0.85), ("broth_chicken", 0.10), ("herbs_fresh", 0.05)],
    "sambal_stingray": [("fish_white", 0.70), ("tomato_sauce", 0.20), ("spices_ground", 0.10)],

    # === PORK & BEEF ===
    "bak_chor_mee": [("wheat_noodles", 0.45), ("pork_lean", 0.25), ("vegetables", 0.20), ("oil_vegetable", 0.10)],
    "bak_kut_teh": [("pork_lean", 0.50), ("broth_chicken", 0.40), ("herbs_fresh", 0.07), ("spices_ground", 0.03)],
    "bak_kwa": [("pork_lean", 0.85), ("sugar_white", 0.10), ("soy_sauce", 0.05)],
    "char_siew": [("pork_lean", 0.80), ("sugar_white", 0.12), ("soy_sauce", 0.08)],
    "char_siew_pau": [("flour_wheat", 0.50), ("pork_lean", 0.35), ("sugar_white", 0.10), ("oil_vegetable", 0.05)],
    "kebab_beef": [("beef_lean", 0.60), ("bread_white", 0.25), ("vegetables", 0.12), ("oil_vegetable", 0.03)],
    "kebab_chicken": [("chicken_breast", 0.60), ("bread_white", 0.25), ("vegetables", 0.12), ("oil_vegetable", 0.03)],
    "kebab___beef": [("beef_lean", 0.60), ("bread_white", 0.25), ("vegetables", 0.12), ("oil_vegetable", 0.03)],
    "kebab___chicken": [("chicken_breast", 0.60), ("bread_white", 0.25), ("vegetables", 0.12), ("oil_vegetable", 0.03)],
    "lamb_chops": [("beef_lean", 0.80), ("oil_olive", 0.15), ("herbs_fresh", 0.05)],
    "sirloin_steak": [("beef_lean", 0.85), ("oil_olive", 0.10), ("herbs_fresh", 0.05)],
    "satay": [("chicken_breast", 0.60), ("peanuts", 0.25), ("spices_ground", 0.10), ("oil_vegetable", 0.05)],

    # === VEGETABLES ===
    "green_leafy_vegetables": [("spinach", 1.0)],
    "mixed_vegetables": [("broccoli", 0.30), ("carrots", 0.30), ("bell_peppers", 0.25), ("onions", 0.15)],
    "sambal_kangkung": [("spinach", 0.70), ("tomato_sauce", 0.20), ("spices_ground", 0.10)],
    "salad": [("lettuce", 0.50), ("tomatoes", 0.20), ("cucumber", 0.15), ("carrots", 0.10), ("oil_olive", 0.05)],
    "baked_beans": [("beans_cooked", 0.80), ("tomato_sauce", 0.15), ("sugar_white", 0.05)],

    # === TOFU & SOY ===
    "tauhu_goreng": [("tofu_firm", 0.70), ("flour_wheat", 0.20), ("oil_vegetable", 0.10)],
    "hotplate_tofu": [("tofu_firm", 0.60), ("vegetables", 0.25), ("oil_vegetable", 0.15)],
    "yong_tau_foo": [("tofu_firm", 0.50), ("fish_white", 0.30), ("vegetables", 0.15), ("broth_vegetable", 0.05)],

    # === PASTA & WESTERN ===
    "macaroni": [("pasta_cooked", 0.70), ("cheese_soft", 0.20), ("milk_whole", 0.10)],
    "pasta_fettuccine": [("pasta_cooked", 0.60), ("cream_heavy", 0.25), ("cheese_hard", 0.15)],
    "pasta_red_sauce": [("pasta_cooked", 0.60), ("tomato_sauce", 0.30), ("cheese_hard", 0.10)],
    "pasta___fettuccine": [("pasta_cooked", 0.60), ("cream_heavy", 0.25), ("cheese_hard", 0.15)],
    "pasta___red_sauce": [("pasta_cooked", 0.60), ("tomato_sauce", 0.30), ("cheese_hard", 0.10)],
    "wholegrain_pasta": [("pasta_cooked", 1.0)],
    "lasagna": [("pasta_cooked", 0.35), ("cheese_soft", 0.25), ("beef_regular", 0.20), ("tomato_sauce", 0.20)],
    "pizza": [("pizza_dough", 0.50), ("tomato_sauce", 0.25), ("cheese_soft", 0.20), ("herbs_fresh", 0.05)],
    "sandwich": [("bread_white", 0.45), ("chicken_breast", 0.25), ("vegetables", 0.25), ("oil_vegetable", 0.05)],
    "french_fries": [("potatoes", 0.70), ("oil_vegetable", 0.30)],
    "cheese_fries": [("potatoes", 0.60), ("cheese_soft", 0.25), ("oil_vegetable", 0.15)],
    "mixed_grills": [("beef_lean", 0.40), ("chicken_breast", 0.35), ("vegetables", 0.20), ("oil_olive", 0.05)],

    # === SNACKS ===
    "chocolate": [("sugar_white", 0.50), ("oil_vegetable", 0.35), ("milk_whole", 0.15)],
    "nuts": [("nuts_mixed", 1.0)],
    "snacks_and_chips": [("potatoes", 0.60), ("oil_vegetable", 0.35), ("spices_ground", 0.05)],
    "sweets": [("sugar_white", 0.80), ("oil_vegetable", 0.15), ("milk_whole", 0.05)],
    "popcorn": [("corn", 0.70), ("oil_vegetable", 0.25), ("spices_ground", 0.05)],
    "preserved_fruit_snacks": [("strawberries", 0.60), ("sugar_white", 0.35), ("herbs_dried", 0.05)],
    "seaweed_snack": [("vegetables", 0.80), ("oil_sesame", 0.15), ("spices_ground", 0.05)],

    # === DESSERTS ===
    "ice_cream_chocolate": [("milk_whole", 0.50), ("cream_heavy", 0.30), ("sugar_white", 0.15), ("egg_whole", 0.05)],
    "ice_cream_vanilla": [("milk_whole", 0.50), ("cream_heavy", 0.30), ("sugar_white", 0.15), ("egg_whole", 0.05)],
    "ice_cream___chocolate": [("milk_whole", 0.50), ("cream_heavy", 0.30), ("sugar_white", 0.15), ("egg_whole", 0.05)],
    "ice_cream___vanilla": [("milk_whole", 0.50), ("cream_heavy", 0.30), ("sugar_white", 0.15), ("egg_whole", 0.05)],
    "ice_kacang": [("water", 0.60), ("sugar_white", 0.25), ("milk_whole", 0.10), ("strawberries", 0.05)],
    "tiramisu": [("cheese_soft", 0.35), ("cream_heavy", 0.30), ("sugar_white", 0.20), ("flour_wheat", 0.15)],
    "parfait": [("yogurt_plain", 0.50), ("strawberries", 0.30), ("oats_cooked", 0.15), ("honey", 0.05)],
    "mango_pudding": [("mango", 0.60), ("milk_whole", 0.25), ("sugar_white", 0.15)],

    # === TRADITIONAL SWEETS ===
    "cny_love_letter": [("flour_wheat", 0.60), ("sugar_white", 0.25), ("oil_vegetable", 0.15)],
    "kueh_salat": [("rice_cooked", 0.50), ("coconut_milk", 0.30), ("sugar_white", 0.15), ("pineapple", 0.05)],
    "peanut_pancake": [("flour_wheat", 0.50), ("peanuts", 0.30), ("sugar_white", 0.15), ("oil_vegetable", 0.05)],
    "pung_kueh": [("flour_wheat", 0.60), ("coconut_milk", 0.25), ("sugar_white", 0.15)],
    "kueh_lapis_rainbow": [("flour_wheat", 0.40), ("coconut_milk", 0.30), ("sugar_white", 0.25), ("oil_vegetable", 0.05)],
    "kueh_lapis___rainbow": [("flour_wheat", 0.40), ("coconut_milk", 0.30), ("sugar_white", 0.25), ("oil_vegetable", 0.05)],
    "kueh_lapis_baked": [("flour_wheat", 0.45), ("egg_whole", 0.25), ("sugar_white", 0.20), ("butter", 0.10)],
    "kuih_bahulu": [("flour_wheat", 0.50), ("egg_whole", 0.30), ("sugar_white", 0.20)],
    "pineapple_tarts": [("flour_wheat", 0.50), ("pineapple", 0.30), ("butter", 0.15), ("sugar_white", 0.05)],
    "tau_suan": [("beans_cooked", 0.60), ("water", 0.30), ("sugar_white", 0.10)],
    "tutu_kueh": [("rice_cooked", 0.60), ("coconut_milk", 0.25), ("sugar_white", 0.15)],
    "egg_tart": [("flour_wheat", 0.40), ("egg_whole", 0.35), ("cream_heavy", 0.20), ("sugar_white", 0.05)],
    "cheng_tng": [("water", 0.70), ("sugar_white", 0.20), ("beans_cooked", 0.10)],

    # === DUMPLINGS & WRAPPED FOODS ===
    "dumpling": [("flour_wheat", 0.50), ("pork_lean", 0.30), ("vegetables", 0.15), ("oil_vegetable", 0.05)],
    "siew_mai": [("pork_lean", 0.50), ("flour_wheat", 0.30), ("vegetables", 0.15), ("oil_vegetable", 0.05)],
    "rice_dumpling": [("rice_cooked", 0.60), ("pork_lean", 0.25), ("vegetables", 0.10), ("oil_vegetable", 0.05)],
    "xiao_long_bao": [("flour_wheat", 0.45), ("pork_lean", 0.35), ("broth_chicken", 0.15), ("vegetables", 0.05)],
    "spring_rolls": [("vegetables", 0.45), ("rice_noodles", 0.25), ("shrimp", 0.20), ("oil_vegetable", 0.10)],
    "popiah": [("flour_wheat", 0.40), ("vegetables", 0.35), ("shrimp", 0.20), ("oil_vegetable", 0.05)],
    "curry_puff": [("flour_wheat", 0.50), ("potatoes", 0.30), ("chicken_breast", 0.15), ("oil_vegetable", 0.05)],
    "chwee_kueh": [("rice_cooked", 0.70), ("pork_lean", 0.20), ("oil_vegetable", 0.10)],

    # === TRADITIONAL DISHES ===
    "chap_chye_nonya": [("vegetables", 0.60), ("tofu_firm", 0.25), ("mushrooms", 0.10), ("oil_vegetable", 0.05)],
    "chinese_fritters": [("flour_wheat", 0.60), ("oil_vegetable", 0.35), ("spices_ground", 0.05)],
    "chai_tow_kuay": [("rice_cooked", 0.60), ("egg_whole", 0.25), ("oil_vegetable", 0.15)],
    "goreng_pisang": [("bananas", 0.60), ("flour_wheat", 0.25), ("oil_vegetable", 0.15)],
    "begedil": [("potatoes", 0.70), ("egg_whole", 0.20), ("oil_vegetable", 0.10)],
    "ngoh_hiang": [("pork_lean", 0.50), ("vegetables", 0.30), ("flour_wheat", 0.15), ("oil_vegetable", 0.05)],
    "otak": [("fish_white", 0.60), ("coconut_milk", 0.25), ("spices_ground", 0.10), ("herbs_fresh", 0.05)],
    "oyster_omelette": [("egg_whole", 0.50), ("shrimp", 0.30), ("flour_wheat", 0.15), ("oil_vegetable", 0.05)],
    "rojak": [("pineapple", 0.30), ("cucumber", 0.25), ("apples", 0.20), ("peanuts", 0.15), ("spices_ground", 0.10)],
    "roti_john": [("bread_white", 0.50), ("egg_whole", 0.30), ("onions", 0.15), ("oil_vegetable", 0.05)],
    "sambal": [("tomatoes", 0.60), ("chili_peppers", 0.25), ("onions", 0.10), ("oil_vegetable", 0.05)],
    "kway_chap": [("rice_noodles", 0.45), ("pork_lean", 0.30), ("broth_chicken", 0.20), ("vegetables", 0.05)],
    "meatball": [("beef_regular", 0.70), ("flour_wheat", 0.20), ("egg_whole", 0.10)],
    "bakso": [("beef_regular", 0.60), ("flour_wheat", 0.25), ("broth_beef", 0.15)],
    "sausage_rolls": [("pork_lean", 0.50), ("flour_wheat", 0.35), ("oil_vegetable", 0.15)],
    "steamed_buns": [("flour_wheat", 0.70), ("pork_lean", 0.20), ("vegetables", 0.08), ("sugar_white", 0.02)],
    "yuseng": [("fish_white", 0.40), ("vegetables", 0.35), ("rice_noodles", 0.20), ("oil_sesame", 0.05)],

    # === INTERNATIONAL ===
    "paella_seafood": [("rice_cooked", 0.40), ("shrimp", 0.25), ("fish_white", 0.20), ("vegetables", 0.10), ("oil_olive", 0.05)],
    "tacos_and_nachos": [("bread_white", 0.40), ("beef_regular", 0.30), ("cheese_soft", 0.20), ("vegetables", 0.10)],
    "tortilla_plain": [("flour_wheat", 0.80), ("oil_vegetable", 0.15), ("water", 0.05)],
    "tortilla___plain": [("flour_wheat", 0.80), ("oil_vegetable", 0.15), ("water", 0.05)],

    # === SPECIAL RICE DISHES ===
    "rice_chicken_katsu_with_japanese_curry": [("rice_cooked", 0.45), ("chicken_breast", 0.30), ("vegetables", 0.20), ("flour_wheat", 0.05)],
    "pilaf_pea": [("rice_cooked", 0.70), ("vegetables", 0.25), ("oil_vegetable", 0.05)],
    "yogurt": [("yogurt_plain", 1.0)],

    # === CANNED/PROCESSED ===
    "canned_fruit_salad": [("pineapple", 0.40), ("apples", 0.30), ("grapes", 0.20), ("sugar_white", 0.10)],
}

def convert_to_underscore_format():
    """Convert the class names to match the underscore format expected by the system."""
    # Convert food class names to underscore format
    from food_class_names import FOOD_CLASS_NAMES

    converted_mappings = {}

    for class_name in FOOD_CLASS_NAMES:
        # Convert to underscore format
        underscore_name = class_name.lower().replace(' ', '_').replace(',', '').replace('-', '_')
        underscore_name = underscore_name.replace('&', 'and').replace('(', '').replace(')', '')

        if underscore_name in COMPLETE_244_MAPPINGS:
            converted_mappings[underscore_name] = COMPLETE_244_MAPPINGS[underscore_name]
        else:
            print(f"Missing mapping for: {underscore_name} (original: {class_name})")

    return converted_mappings

if __name__ == "__main__":
    # Test the conversion
    converted = convert_to_underscore_format()
    print(f"Mapped {len(converted)} out of 244 food classes")
    print(f"Coverage: {len(converted)/244*100:.1f}%")