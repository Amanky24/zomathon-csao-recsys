import pandas as pd
import numpy as np
import os

# 1. Users Schema
users_schema = pd.DataFrame({
    'user_id': pd.Series(dtype='str'),
    'user_segment': pd.Series(dtype='category'),
    'order_frequency': pd.Series(dtype='category'),
    'dietary_preference': pd.Series(dtype='category'),
    'discount_reliance': pd.Series(dtype='bool'),
    'aov_bracket': pd.Series(dtype='category'),
    'affinity_scores': pd.Series(dtype='object'), # JSON/Dict stored as object
    'exploration_index': pd.Series(dtype='float'),
    'user_demographic_cluster': pd.Series(dtype='category')
})

# 2. Restaurants Schema
restaurants_schema = pd.DataFrame({
    'restaurant_id': pd.Series(dtype='str'),
    'restaurant_type': pd.Series(dtype='category'),
    'primary_cuisine': pd.Series(dtype='category'),
    'avg_cost_for_two': pd.Series(dtype='float')
})

# 3. Items (Catalog) Schema
items_schema = pd.DataFrame({
    'item_id': pd.Series(dtype='str'),
    'restaurant_id': pd.Series(dtype='str'),
    'item_name': pd.Series(dtype='str'),
    'macro_category': pd.Series(dtype='category'),
    'price': pd.Series(dtype='float'),
    'is_veg': pd.Series(dtype='bool'),
    'prep_time_tier': pd.Series(dtype='category'),
    'portion_size': pd.Series(dtype='category'),
    'allergen_tags': pd.Series(dtype='object') # Array stored as object
})

# 4. Interactions Schema (Combines Session, Cart State, and Target) 
interactions_schema = pd.DataFrame({
    # Session Details
    'session_id': pd.Series(dtype='str'),
    'user_id': pd.Series(dtype='str'),
    'restaurant_id': pd.Series(dtype='str'),
    'time_of_day': pd.Series(dtype='category'),
    'is_weekend': pd.Series(dtype='bool'),
    'delivery_zone_type': pd.Series(dtype='category'),
    
    # Cart State
    'current_cart_value_inr': pd.Series(dtype='float'),
    'current_item_count': pd.Series(dtype='int'),
    'cart_dietary_flag': pd.Series(dtype='category'),
    'has_main_course': pd.Series(dtype='bool'),
    'has_beverage': pd.Series(dtype='bool'),
    'has_dessert': pd.Series(dtype='bool'),
    'has_starter_or_side': pd.Series(dtype='bool'),
    'dominant_cuisine': pd.Series(dtype='category'),
    
    # Target / Output
    'candidate_item_id': pd.Series(dtype='str'),
    'candidate_macro_cat': pd.Series(dtype='category'),
    'candidate_price_ratio': pd.Series(dtype='float'),
    'was_accepted': pd.Series(dtype='int8') # 1 or 0 for ML Label
})

# Export to the raw data folder
os.makedirs('../data/raw', exist_ok=True) 

users_schema.to_csv('../data/raw/users_schema.csv', index=False)
restaurants_schema.to_csv('../data/raw/restaurants_schema.csv', index=False)
items_schema.to_csv('../data/raw/items_schema.csv', index=False)
interactions_schema.to_csv('../data/raw/interactions_schema.csv', index=False)

print("Success! Updated schema CSVs generated in data/raw/")