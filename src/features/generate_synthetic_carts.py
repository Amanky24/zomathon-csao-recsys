import pandas as pd
import numpy as np

class CSAODataSimulator:
    """
    Stochastic data generation engine for the Cart Super Add-On (CSAO) Recommendation System.
    Models user cart-building trajectories using conditional probabilities, transition matrices, 
    and domain-specific heuristics to generate synthetic, high-fidelity interaction logs.
    """
    def __init__(self):
        # Cross-Category Affinity Matrix
        # Represents the baseline transition probability (modifier) of adding a candidate category 
        # given the current dominant category in the cart state.
        self.base_matrix = {
            'Main_Course':   {'Main_Course': -0.05, 'Starter': 0.02,  'Beverage': 0.10,  'Dessert': 0.05, 'Accompaniment': 0.15},
            'Starter':       {'Main_Course': 0.05,  'Starter': -0.02, 'Beverage': 0.12,  'Dessert': 0.02, 'Accompaniment': 0.10},
            'Beverage':      {'Main_Course': 0.00,  'Starter': 0.10,  'Beverage': -0.05, 'Dessert': 0.08, 'Accompaniment': 0.00},
            'Dessert':       {'Main_Course': -0.05, 'Starter': -0.05, 'Beverage': 0.05,  'Dessert': 0.02, 'Accompaniment': -0.05},
            'Accompaniment': {'Main_Course': 0.10,  'Starter': 0.05,  'Beverage': 0.05,  'Dessert': 0.00, 'Accompaniment': -0.05}
        }

    def calculate_probability(self, cart, candidate, user, context):
        """
        Calculates the conditional probability of a candidate item being accepted into the cart.
        Ingests the current cart state, candidate features, user profile, and session context.
        """
        
        # 1. Initialize with Global Baseline
        dominant_cat = cart['dominant_cuisine'] 
        candidate_cat = candidate['macro_category']
        
        # Base probability derived from the item's historical global attach rate [cite: 118]
        prob = candidate['restaurant_item_attach']
        
        # Apply cross-category affinity modifier if a valid transition exists
        if dominant_cat in self.base_matrix and candidate_cat in self.base_matrix[dominant_cat]:
            prob += self.base_matrix[dominant_cat][candidate_cat]

        # 2. Apply Deterministic Business Rules & Heuristics
        
        # --- Price & Promotion Sensitivity ---
        if (candidate['price'] / cart['current_cart_value_inr']) > 0.25:
            prob = 0.02  # The Price Wall Constraint
            
        if user['user_segment'] == 'Budget' and candidate['price'] > 99:
            prob = 0.05  # Budget Cap Hard Limit
            
        if user['discount_reliance'] == True and candidate['macro_category'] == 'Accompaniment/Dip':
            prob += 0.20 # Discount Chaser Affinity

        # --- Cart Composition & Sequential Dependencies ---
        if cart['has_main_course'] and cart['has_beverage'] and not cart['has_dessert']: 
            if candidate['macro_category'] == 'Dessert': 
                prob = 0.40 # Meal Finisher Completion Logic
                
        if context['restaurant_type'] == 'QSR/Fast_Food' and not cart['has_starter_or_side']:
            if candidate['macro_category'] == 'Starter':
                prob = 0.60 # QSR Structural Baseline
                
        if cart['time_since_last_add'] > 45 and candidate['price'] < 50: 
            prob += 0.25 # Staller/Hesitation Conversion Trigger

        # --- Spatio-Temporal & Environmental Context ---
        if context['weather_proxy'] == 'Rainy/Cold' and candidate['temperature_state'] == 'Hot':
            prob += 0.30 # Weather-Driven Comfort Affinity
            
        if context['delivery_zone_type'] == 'Office' and context['time_of_day'] == 'Lunch' and candidate['prep_time_tier'] == 'Fast':
            prob += 0.20 # Office Lunch Velocity Requirement
            
        if context['is_weekend'] == True and candidate['health_index'] == 'Indulgent':
            prob = 0.35 # Weekend Indulgence Override

        # --- Dietary & Health Constraints ---
        if cart['cart_avg_health_index'] == 'Low_Calorie' and candidate['health_index'] == 'Indulgent':
            prob = 0.01 # Health Halo Suppression
            
        if cart['cart_dietary_flag'] == 'Veg' and candidate['is_veg'] == False:
            prob = 0.00 # Strict Dietary Wall (Absolute Block)

        # --- Fulfillment & Operational Constraints ---
        if context['transit_distance_km'] > 7 and candidate['temperature_state'] == 'Cold':
            prob = 0.02 # Melt Risk Distance Suppression
            
        if context['restaurant_busyness'] > 8 and candidate['prep_time_tier'] == 'Slow':
            prob = 0.00 # Kitchen Load-Balancing Block

        # --- Social Proof & Order Scale ---
        if cart['current_item_count'] == 1 and candidate['portion_size'] == 'Family_Pack':
            prob = 0.01 # Single-User Portion Mismatch
            
        if cart['current_item_count'] > 3 and candidate['portion_size'] == 'Sharing':
            prob += 0.25 # Multi-User Sharing Affinity
            
        if cart['item_trending_score'] > 80:
            prob += 0.15 # Local Trending / Virality Boost

        # 3. Output Bound Enforcement
        # Ensure final probability remains strictly within [0.0, 1.0] mathematical bounds
        return max(0.0, min(1.0, prob))

    def simulate_decision(self, probability):
        """
        Executes a stochastic trial against the computed probability.
        Returns 1 (Accepted) or 0 (Rejected) for the target variable mapping.
        """
        return 1 if np.random.random() < probability else 0

# --- Validation Initialization ---
simulator = CSAODataSimulator()
print("CSAO Simulation Engine successfully initialized with operational constraints.")