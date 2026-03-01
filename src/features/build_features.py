import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class FeatureEngineer:
    """
    Feature Engineering Pipeline for the CSAO Recommendation System.
    Transforms raw synthetic cart logs into a numerical feature matrix 
    optimized for the Stage 2 XGBoost/LightGBM Ranker.
    """
    def __init__(self):
        self.encoders = {}
        
        # 1. Categorical Columns (Sourced directly from generate_dataset.py)
        self.cat_cols = [
            'user_demographic_cluster', 
            'user_segment', 
            'time_of_day', 
            'weather_proxy', 
            'dominant_cuisine', 
            'candidate_macro_cat'
        ]
        
        # 2. Numerical Columns
        self.num_cols = [
            'transit_distance_km', 
            'restaurant_busyness', 
            'current_cart_value_inr', 
            'candidate_price_ratio'
        ]
        
        # 3. Boolean Columns
        self.bool_cols = ['is_weekend']

    def fit_transform(self, df):
        """
        Fits the categorical encoders on the training dataset and transforms it.
        Returns the Feature Matrix (X) and the Target Array (y).
        """
        processed_df = df.copy()
        
        # Encode Categoricals
        for col in self.cat_cols:
            le = LabelEncoder()
            # Convert to string safely to avoid mixed-type errors
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            self.encoders[col] = le
            
        # Convert Booleans to Integers (True=1, False=0)
        for col in self.bool_cols:
            processed_df[col] = processed_df[col].astype(int)
            
        # Assemble final feature list
        feature_cols = self.cat_cols + self.num_cols + self.bool_cols
        X = processed_df[feature_cols]
        
        # Extract target variable if it exists (it will during training)
        if 'was_accepted' in processed_df.columns:
            y = processed_df['was_accepted'].astype(int)
            return X, y
            
        return X

    def transform(self, df):
        """
        Transforms live inference data using the PRE-FITTED encoders.
        This ensures O(1) encoding time to meet the sub-300ms latency constraint.
        """
        if not self.encoders:
            raise ValueError("Encoders are not fitted! Call fit_transform() first.")
            
        processed_df = df.copy()
        
        for col in self.cat_cols:
            # Use .transform() to apply the exact mapping learned during training
            processed_df[col] = self.encoders[col].transform(processed_df[col].astype(str))
            
        for col in self.bool_cols:
            processed_df[col] = processed_df[col].astype(int)
            
        feature_cols = self.cat_cols + self.num_cols + self.bool_cols
        return processed_df[feature_cols]

    def save_encoders(self, filepath='src/models/checkpoints/feature_encoders.pkl'):
        """Saves the fitted encoders so the API Gateway can use them in production."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.encoders, filepath)
        print(f"Feature encoders successfully saved to {filepath}")
        
    def load_encoders(self, filepath='src/models/checkpoints/feature_encoders.pkl'):
        """Loads fitted encoders into memory."""
        self.encoders = joblib.load(filepath)