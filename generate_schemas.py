import pandas as pd
import numpy as np
import os

# 1. Define the schemas
users_schema = pd.DataFrame({
    'user_id': pd.Series(dtype='str'),
    'order_frequency': pd.Series(dtype='float'), 
    'recency_days': pd.Series(dtype='int'),      
    'monetary_value': pd.Series(dtype='float'),  
    'preferred_zone': pd.Series(dtype='category'),
    'user_segment': pd.Series(dtype='category')  
})

restaurants_schema = pd.DataFrame({
    'restaurant_id': pd.Series(dtype='str'),
    'cuisine_type': pd.Series(dtype='category'),
    'price_range': pd.Series(dtype='category'),
    'rating': pd.Series(dtype='float'),
    'delivery_rating': pd.Series(dtype='float'),
    'order_volume': pd.Series(dtype='int'),
    'is_chain': pd.Series(dtype='bool')
})

items_schema = pd.DataFrame({
    'item_id': pd.Series(dtype='str'),
    'restaurant_id': pd.Series(dtype='str'),
    'category': pd.Series(dtype='category'),     
    'is_veg': pd.Series(dtype='bool'),
    'price': pd.Series(dtype='float')
})

interactions_schema = pd.DataFrame({
    'session_id': pd.Series(dtype='str'),
    'user_id': pd.Series(dtype='str'),
    'restaurant_id': pd.Series(dtype='str'),
    'current_cart_items': pd.Series(dtype='object'), 
    'cart_value': pd.Series(dtype='float'),          
    'timestamp': pd.Series(dtype='datetime64[ns]'),
    'day_of_week': pd.Series(dtype='category'),      
    'meal_time': pd.Series(dtype='category'),        
    'candidate_item_id': pd.Series(dtype='str'),     
    'is_accepted': pd.Series(dtype='int8')           
})

# 2. Export to the raw data folder
users_schema.to_csv('data/raw/users_schema.csv', index=False)
restaurants_schema.to_csv('data/raw/restaurants_schema.csv', index=False)
items_schema.to_csv('data/raw/items_schema.csv', index=False)
interactions_schema.to_csv('data/raw/interactions_schema.csv', index=False)

print("Success! Empty schema CSVs generated in data/raw/")