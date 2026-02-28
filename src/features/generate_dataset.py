import pandas as pd
import numpy as np
import uuid

# 1. Load the exact PFC Menu and the Simulator
menu_df = pd.read_csv('data/raw/pfc_menu.csv')

# Ensure column names map perfectly to the simulator's logic
menu_df.rename(columns={'item name': 'item_name', 'flavor profile': 'flavor_profile', 
                        'temperature state': 'temperature_state'}, inplace=True)

from generate_synthetic_carts import CSAODataSimulator
simulator = CSAODataSimulator()

interactions = []

# 2. Strict category lists defined by Baqir's Parameter List
user_segments = ['Budget', 'Mid-Tier', 'Premium']
weather_proxies = ['Normal', 'Rainy/Cold']
time_of_day_opts = ['Breakfast', 'Lunch', 'Snack', 'Dinner', 'Late_Night']
delivery_zones = ['Office/Commercial', 'Residential/Home']

print("Spinning up 10,000 cart sessions. Please wait...")

# 3. The Monte Carlo Loop
for i in range(10000):
    # A. Sample strictly from the allowed parameter bounds
    user = {
        'user_id': f"U_{np.random.randint(1000, 9999)}",
        'user_segment': np.random.choice(user_segments),
        'discount_reliance': np.random.choice([True, False])
    }
    
    context = {
        'session_id': str(uuid.uuid4()),
        'restaurant_id': 'PFC', 
        'restaurant_type': 'QSR/Fast_Food', 
        'time_of_day': np.random.choice(time_of_day_opts),
        'is_weekend': np.random.choice([True, False]),
        'weather_proxy': np.random.choice(weather_proxies),
        'delivery_zone_type': np.random.choice(delivery_zones),
        'transit_distance_km': np.round(np.random.uniform(1.0, 10.0), 1),
        'restaurant_busyness': np.random.randint(1, 11)
    }
    
    # B. Build the Cart (Sample 1 to 4 items from the PFC Menu)
    cart_items = menu_df.sample(n=np.random.randint(1, 5))
    
    # C. Calculate exact Cart Aggregates
    cart = {
        'current_cart_value_inr': cart_items['price'].sum(),
        'current_item_count': len(cart_items),
        'has_main_course': 'Main Course' in cart_items['macro_category'].values,
        'has_beverage': 'Beverage' in cart_items['macro_category'].values,
        'has_dessert': 'Dessert' in cart_items['macro_category'].values,
        'has_starter_or_side': ('Starter' in cart_items['macro_category'].values) or ('Accompaniment' in cart_items['macro_category'].values),
        'dominant_cuisine': cart_items['macro_category'].mode()[0], 
        'cart_dietary_flag': 'Veg' if cart_items['is_veg'].all() == 1 else 'Mixed',
        'cart_avg_health_index': cart_items['health_index'].mode()[0] if not cart_items['health_index'].isnull().all() else 'Unknown',
        'time_since_last_add': np.random.randint(10, 120),
        'item_trending_score': np.random.randint(10, 100)
    }
    
    # D. Pick a Candidate Item (Must be an item NOT already in the cart)
    remaining_menu = menu_df[~menu_df['item_id'].isin(cart_items['item_id'])]
    candidate = remaining_menu.sample(1).iloc[0].to_dict()
    
    # E. Run the Simulator Logic
    probability = simulator.calculate_probability(cart, candidate, user, context)
    was_accepted = simulator.simulate_decision(probability)
    
    # F. Record the Final Interaction
    interaction_record = {
        'session_id': context['session_id'],
        'user_id': user['user_id'],
        'restaurant_id': context['restaurant_id'],
        'time_of_day': context['time_of_day'],
        'is_weekend': context['is_weekend'],
        'weather_proxy': context['weather_proxy'],
        'delivery_zone_type': context['delivery_zone_type'],
        'transit_distance_km': context['transit_distance_km'],
        'restaurant_busyness': context['restaurant_busyness'],
        'current_cart_value_inr': cart['current_cart_value_inr'],
        'current_item_count': cart['current_item_count'],
        'cart_dietary_flag': cart['cart_dietary_flag'],
        'has_main_course': cart['has_main_course'],
        'has_beverage': cart['has_beverage'],
        'has_dessert': cart['has_dessert'],
        'has_starter_or_side': cart['has_starter_or_side'],
        'dominant_cuisine': cart['dominant_cuisine'],
        'candidate_item_id': candidate['item_id'],
        'candidate_macro_cat': candidate['macro_category'],
        'candidate_price_ratio': candidate['price'] / cart['current_cart_value_inr'] if cart['current_cart_value_inr'] > 0 else 0,
        'was_accepted': was_accepted
    }
    
    interactions.append(interaction_record)

# 4. Save the final Dataset to the Processed folder
final_dataset = pd.DataFrame(interactions)
final_dataset.to_csv('data/processed/synthetic_interactions_10k.csv', index=False)
print(f"Success! Generated {len(final_dataset)} realistic cart sessions in data/processed/")