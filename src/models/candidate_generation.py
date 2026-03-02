import pandas as pd
import numpy as np
import os


from src.features.generate_synthetic_carts import PERSONA_CONFIG

class CandidateGenerator:
    """
    Candidate Generation (Stage 1): High-speed retrieval of potential add-ons.
    Filters the PFC menu down to a relevant subset based on co-occurrence and personas.
    """
    def __init__(self, interaction_data_path, menu_data_path):
        
        self.interactions_df = pd.read_csv(interaction_data_path)
        self.menu_df = pd.read_csv(menu_data_path)
        
        
        self.menu_df.rename(columns={
            'item name': 'item_name', 
            'flavor profile': 'flavor_profile', 
            'temperature state': 'temperature_state'
        }, inplace=True, errors='ignore')
        
        self.co_occurrence_map = self._build_co_occurrence_map()

    def _build_co_occurrence_map(self):
        """
        Identifies which items are accepted most frequently given a dominant cuisine.
        Matches 'dominant_cuisine' and 'candidate_item_id' from the CSV schema.
        """
        # Only learn from successful user conversions
        success_interactions = self.interactions_df[self.interactions_df['was_accepted'] == 1]
        
        # Map: Dominant Category -> List of Top 20 Accepted Item IDs
        return success_interactions.groupby('dominant_cuisine')['candidate_item_id'].apply(
            lambda x: x.value_counts().head(20).index.tolist()
        ).to_dict()

    def get_candidates(self, cart_state, persona_name, top_k=15):
        """
        Retrieves candidates and applies the strict 'Hard Rejections' from your PERSONA_CONFIG.
        """
        dominant_cat = cart_state.get('dominant_cuisine')
        persona_rules = PERSONA_CONFIG.get(persona_name, {})
        
        # 1. RETRIEVAL: Use Co-occurrence Map; Fallback to global attach rates
        candidate_ids = self.co_occurrence_map.get(dominant_cat, [])
        if not candidate_ids:
            candidate_ids = self.menu_df.sort_values(
                by='restaurant_item_attach', ascending=False
            )['item_id'].head(top_k).tolist()

        # Load item details from menu
        potential_items = self.menu_df[self.menu_df['item_id'].isin(candidate_ids)].to_dict('records')

        # 2. FILTERING: Apply Persona Engine "Hard Rejections" 
        filtered_candidates = []
        rejects = persona_rules.get('rejects', [])

        for item in potential_items:
            # Enforce 'Strict Veg' logic 
            if 'Non-Veg' in rejects and item['is_veg'] == 0:
                continue
            
            # Enforce 'Late-Night/Corporate' speed logic 
            if 'Slow' in rejects and item['prep_time_tier'] == 'Slow':
                continue
                
            # Enforce category-specific hates (e.g. Protein Junkie rejects Desserts) 
            if item['macro_category'] in rejects:
                continue
            
            # Enforce portion size constraints (e.g. Party Host rejects Single-Serve) 
            if 'Single-Serve' in rejects and item['portion_size'] == 'Single_Serve':
                continue

            filtered_candidates.append(item)

        return filtered_candidates[:top_k]
