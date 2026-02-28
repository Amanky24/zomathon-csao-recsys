import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

class FeatureEngineer:
    def __init__(self):
        self.encoders = {}
        # Core categories defined in your Zomathon Data Sheet
        self.cat_cols = [
            'user_demographic_cluster', 'user_segment', 'time_of_day', 
            'weather_proxy', 'dominant_cuisine', 'candidate_macro_cat'
        ]
        self.num_cols = ['current_cart_value_inr', 'transit_distance_km', 'candidate_price_ratio']

    def transform(self, df, is_training=True):
        processed_df = df.copy()

        for col in self.cat_cols:
            if is_training:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                self.encoders[col] = le
            else:
                processed_df[col] = self.encoders[col].transform(processed_df[col].astype(str))

        features = self.cat_cols + self.num_cols
        if 'was_accepted' in processed_df.columns:
            return processed_df[features], processed_df['was_accepted']
        return processed_df[features]