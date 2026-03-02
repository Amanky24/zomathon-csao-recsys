import pandas as pd
import numpy as np

PERSONA_CONFIG = {
    'The Strict Veg Family': {
        'segment': 'Premium', 'dietary': 'Veg', 'price_tolerance': 0.35,
        'boosts': {'Accompaniment': 0.15, 'Dessert': 0.10},
        'rejects': ['Non-Veg']
    },
    'Late-Night Solo Binge': {
        'segment': 'Budget', 'dietary': 'Mixed', 'price_tolerance': 0.15,
        'boosts': {'Accompaniment/Dip': 0.30, 'Beverage': 0.15},
        'rejects': ['Slow', 'Main_Course']
    },
    'The Protein Junkie': {
        'segment': 'Premium', 'dietary': 'Non-Veg', 'price_tolerance': 0.25,
        'boosts': {'Starter': 0.20, 'Soup': 0.15},
        'rejects': ['Dessert', 'Bread']
    },
    'Weekend Party Host': {
        'segment': 'Premium', 'dietary': 'Mixed', 'price_tolerance': 0.50,
        'boosts': {'Beverage': 0.20, 'Starter': 0.15},
        'rejects': ['Single-Serve']
    },
    'The Corporate Luncher': {
        'segment': 'Budget', 'dietary': 'Mixed', 'price_tolerance': 0.20,
        'boosts': {'Beverage': 0.10},
        'rejects': ['Slow', 'Dessert']
    },
    'The Discount Chaser': {
        'segment': 'Budget', 'dietary': 'Mixed', 'price_tolerance': 0.0,
        'boosts': {'Accompaniment/Dip': 0.25, 'Accompaniment': 0.10},
        'rejects': ['High-Price'] 
    },
    'The Sweet Tooth': {
        'segment': 'Mid-Tier', 'dietary': 'Mixed', 'price_tolerance': 0.25,
        'boosts': {'Dessert': 0.35, 'Beverage': 0.05},
        'rejects': ['Soup', 'Starter']
    },
    'The Carb Loader': {
        'segment': 'Mid-Tier', 'dietary': 'Veg', 'price_tolerance': 0.25,
        'boosts': {'Accompaniment': 0.20, 'Beverage': 0.10},
        'rejects': ['Soup']
    },
    'The Spice Fiend': {
        'segment': 'Mid-Tier', 'dietary': 'Mixed', 'price_tolerance': 0.25,
        'boosts': {'Beverage': 0.25, 'Accompaniment/Dip': 0.15},
        'rejects': ['Sweet']
    },
    'The Comfort Eater': {
        'segment': 'Mid-Tier', 'dietary': 'Veg', 'price_tolerance': 0.25,
        'boosts': {'Accompaniment': 0.15, 'Soup': 0.10},
        'rejects': ['Spicy', 'Fast Food']
    },
    'The Beverage Addict': {
        'segment': 'Premium', 'dietary': 'Mixed', 'price_tolerance': 0.35,
        'boosts': {'Beverage': 0.40},
        'rejects': ['Accompaniment/Dip', 'Soup']
    },
    'The Strictly Meat Solo': {
        'segment': 'Mid-Tier', 'dietary': 'Non-Veg', 'price_tolerance': 0.25,
        'boosts': {'Starter': 0.15, 'Beverage': 0.10},
        'rejects': ['Veg Mains', 'Dessert']
    },
    'The Indian Gravy Saver': {
        'segment': 'Budget', 'dietary': 'Mixed', 'price_tolerance': 0.20,
        'boosts': {'Accompaniment': 0.35},
        'rejects': ['Starter', 'Dessert']
    },
    'The "Just a Snack" User': {
        'segment': 'Budget', 'dietary': 'Mixed', 'price_tolerance': 0.10,
        'boosts': {'Accompaniment/Dip': 0.20, 'Beverage': 0.10},
        'rejects': ['Main_Course', 'High-Price']
    },
    'The Experimental Foodie': {
        'segment': 'Premium', 'dietary': 'Mixed', 'price_tolerance': 0.40,
        'boosts': {'Dessert': 0.15, 'Starter': 0.15},
        'rejects': ['Basic Accompaniments']
    }
}

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
        Calculates the conditional probability using strict Persona mapping and Context.
        """
        # 1. Initializing with Global Baseline Matrix
        dominant_cat = cart['dominant_cuisine'] 
        candidate_cat = candidate['macro_category']
        prob = candidate['restaurant_item_attach']
        
        if dominant_cat in self.base_matrix and candidate_cat in self.base_matrix[dominant_cat]:
            prob += self.base_matrix[dominant_cat][candidate_cat]

        
        persona_name = user['user_demographic_cluster']
        
        # If the user has a known persona,
        if persona_name in PERSONA_CONFIG:
            persona_rules = PERSONA_CONFIG[persona_name]
            
            # A. Price Tolerance Logic
            cart_val = max(cart['current_cart_value_inr'], 1) # Prevent divide-by-zero
            if (candidate['price'] / cart_val) > persona_rules['price_tolerance']:
                prob = 0.02
                
            # B. Affinity Boosts
            if candidate_cat in persona_rules['boosts']:
                prob += persona_rules['boosts'][candidate_cat]
                
            # C. Hard Rejections (The 0.0 Blocks)
            rejects = persona_rules['rejects']
            
            if 'Non-Veg' in rejects and candidate['is_veg'] == False:
                prob = 0.0 # Strict Veg block
            elif 'Slow' in rejects and candidate['prep_time_tier'] == 'Slow':
                prob = 0.0 # Impatient user block
            elif candidate_cat in rejects: 
                prob = 0.0 # General category block (e.g., hates Soup)
            elif 'Single-Serve' in rejects and candidate['portion_size'] == 'Single':
                prob = 0.0 # Party host block

        # 3. Apply Remaining Spatio-Temporal & Logistics Rules
        
        if context['weather_proxy'] == 'Rainy/Cold' and candidate['temperature_state'] == 'Hot':
            prob += 0.30 
            
        if context['delivery_zone_type'] == 'Office' and context['time_of_day'] == 'Lunch' and candidate['prep_time_tier'] == 'Fast':
            prob += 0.20 
            
        if context['transit_distance_km'] > 7 and candidate['temperature_state'] == 'Cold':
            prob = 0.02 # Melt Risk
            
        if context['restaurant_busyness'] > 8 and candidate['prep_time_tier'] == 'Slow':
            prob = 0.00 # Kitchen Load-Balancing Block

        # 4. Output Bound Enforcement
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