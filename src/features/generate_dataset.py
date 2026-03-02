import pandas as pd
import numpy as np
import uuid

# 1. Load the exact PFC Menu and the Upgraded Simulator
menu_df = pd.read_csv('data/raw/pfc_menu.csv')

# Ensure column names map perfectly to the simulator's logic
menu_df.rename(columns={'item name': 'item_name', 'flavor profile': 'flavor_profile', 
                        'temperature state': 'temperature_state'}, inplace=True)

# Import both the Simulator AND the new Persona Dictionary
from generate_synthetic_carts import CSAODataSimulator, PERSONA_CONFIG
simulator = CSAODataSimulator()

interactions = []

# Context boundaries
weather_proxies = ['Normal', 'Rainy/Cold']
time_of_day_opts = ['Breakfast', 'Lunch', 'Snack', 'Dinner', 'Late_Night']
delivery_zones = ['Office/Commercial', 'Residential/Home']

# Extract the list of 15 Persona Names
persona_names = list(PERSONA_CONFIG.keys())


NUM_SESSIONS = 2000
CANDIDATES_PER_SESSION = 5

print(f"Spinning up {NUM_SESSIONS} cart sessions with {CANDIDATES_PER_SESSION} candidates each. Please wait...")

# 2. The Monte Carlo Loop 
for i in range(NUM_SESSIONS):
    
    # Generating ONE Session ID for this entire group of 5 candidates
    current_session_id = str(uuid.uuid4())
    
    # A. The Persona Injection (Once per session)
    chosen_persona = np.random.choice(persona_names)
    persona_rules = PERSONA_CONFIG[chosen_persona]
    
    user = {
        'user_id': f"U_{np.random.randint(1000, 9999)}",
        'user_demographic_cluster': chosen_persona,       
        'user_segment': persona_rules['segment'],         
        'discount_reliance': True if chosen_persona == 'The Discount Chaser' else np.random.choice([True, False])
    }
    
    # B. Generate the Environment Context (Once per session)
    context = {
        'session_id': current_session_id, # <-- Using the SHARED ID
        'restaurant_id': 'PFC', 
        'restaurant_type': 'QSR/Fast_Food', 
        'time_of_day': np.random.choice(time_of_day_opts),
        'is_weekend': np.random.choice([True, False]),
        'weather_proxy': np.random.choice(weather_proxies),
        'delivery_zone_type': np.random.choice(delivery_zones),
        'transit_distance_km': np.round(np.random.uniform(1.0, 10.0), 1),
        'restaurant_busyness': np.random.randint(1, 11)
    }
    
    # C. Build the Cart (Once per session)
    cart_items = menu_df.sample(n=np.random.randint(1, 5))
    
    cart = {
        'current_cart_value_inr': cart_items['price'].sum(),
        'current_item_count': len(cart_items),
        'has_main_course': 'Main_Course' in cart_items['macro_category'].values,
        'has_beverage': 'Beverage' in cart_items['macro_category'].values,
        'has_dessert': 'Dessert' in cart_items['macro_category'].values,
        'has_starter_or_side': ('Starter' in cart_items['macro_category'].values) or ('Accompaniment' in cart_items['macro_category'].values),
        'dominant_cuisine': cart_items['macro_category'].mode()[0], 
        'cart_dietary_flag': 'Veg' if cart_items['is_veg'].all() == 1 else 'Mixed',
        'cart_avg_health_index': cart_items['health_index'].mode()[0] if not cart_items['health_index'].isnull().all() else 'Unknown',
        'time_since_last_add': np.random.randint(10, 120),
        'item_trending_score': np.random.randint(10, 100)
    }
    
    # D. Pick MULTIPLE Candidate Items
    remaining_menu = menu_df[~menu_df['item_id'].isin(cart_items['item_id'])]
    num_to_sample = min(CANDIDATES_PER_SESSION, len(remaining_menu))
    candidates_df = remaining_menu.sample(num_to_sample)
    
    # --- E. Evaluates all 5 items for this 1 session ---
    for _, candidate_row in candidates_df.iterrows():
        candidate = candidate_row.to_dict()
        
        # Upgraded Persona Logic Simulator
        probability = simulator.calculate_probability(cart, candidate, user, context)
        was_accepted = simulator.simulate_decision(probability)
        
        # F. Recording the Final Interaction 
        interaction_record = {
            'session_id': current_session_id,
            'user_id': user['user_id'],
            'user_demographic_cluster': user['user_demographic_cluster'], 
            'user_segment': user['user_segment'],
            'time_of_day': context['time_of_day'],
            'is_weekend': context['is_weekend'],
            'weather_proxy': context['weather_proxy'],
            'transit_distance_km': context['transit_distance_km'],
            'restaurant_busyness': context['restaurant_busyness'],
            'current_cart_value_inr': cart['current_cart_value_inr'],
            'dominant_cuisine': cart['dominant_cuisine'],
            'candidate_item_id': candidate['item_id'],
            'candidate_macro_cat': candidate['macro_category'],
            'candidate_price_ratio': candidate['price'] / max(cart['current_cart_value_inr'], 1),
            'was_accepted': was_accepted
        }
        
        interactions.append(interaction_record)
    

# 3. Saving the Dataset
final_dataset = pd.DataFrame(interactions)
final_dataset.to_csv('data/processed/synthetic_interactions_10k.csv', index=False)
print(f"Success! Generated {len(final_dataset)} interactions mapped to {NUM_SESSIONS} unique sessions.")