import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class FeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.cat_cols = ['user_demographic_cluster', 'user_segment', 'time_of_day', 
                         'weather_proxy', 'dominant_cuisine', 'candidate_macro_cat']
        self.num_cols = ['transit_distance_km', 'restaurant_busyness', 
                         'current_cart_value_inr', 'candidate_price_ratio']
        self.bool_cols = ['is_weekend']

    def fit_transform(self, df):
        processed_df = df.copy()
        for col in self.cat_cols:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            self.encoders[col] = le
            
        for col in self.bool_cols:
            processed_df[col] = processed_df[col].astype(int)
            
        feature_cols = self.cat_cols + self.num_cols + self.bool_cols
        X = processed_df[feature_cols]
        
        if 'was_accepted' in processed_df.columns:
            y = processed_df['was_accepted'].astype(int)
            return X, y
        return X

    def transform(self, df):
        processed_df = df.copy()
        for col in self.cat_cols:
            processed_df[col] = self.encoders[col].transform(processed_df[col].astype(str))
        for col in self.bool_cols:
            processed_df[col] = processed_df[col].astype(int)
        feature_cols = self.cat_cols + self.num_cols + self.bool_cols
        return processed_df[feature_cols]

    def save_encoders(self, filepath='src/models/checkpoints/feature_encoders.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.encoders, filepath)
        print(f"Feature encoders saved to {filepath}")