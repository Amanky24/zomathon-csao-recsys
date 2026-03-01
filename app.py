from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import joblib

# 1. Initialize the API
app = FastAPI(title="CSAO Recommendation Engine API")

# 2. Load the Model and Encoders on Startup
print("Loading Model and Encoders...")
ranker_model = xgb.XGBRanker()
ranker_model.load_model('src/models/checkpoints/xgb_ranker.json')
feature_encoders = joblib.load('src/models/checkpoints/feature_encoders.pkl')

print("Loading Menu Dictionary...")
menu_df = pd.read_csv('data/raw/pfc_menu.csv')
# Catch both 'item name' and 'item_name' just in case
if 'item name' in menu_df.columns:
    menu_df.rename(columns={'item name': 'item_name'}, inplace=True)

menu_df['item_id'] = menu_df['item_id'].astype(str).str.strip().str.lower()
menu_df['item_name'] = menu_df['item_name'].astype(str).str.strip() 
# Create a fast lookup dictionary: {'pfc_001': 'Classic Burger', ...}
item_id_to_name = dict(zip(menu_df['item_id'], menu_df['item_name']))
# -----------------------------------

# 3. Define the Expected JSON Payload Structure
class CandidateItem(BaseModel):
    candidate_item_id: str
    candidate_macro_cat: str
    candidate_price_ratio: float

class RecommendationRequest(BaseModel):
    session_id: str
    user_id: str
    user_demographic_cluster: str
    user_segment: str
    time_of_day: str
    is_weekend: bool
    weather_proxy: str
    transit_distance_km: float
    restaurant_busyness: int
    current_cart_value_inr: float
    dominant_cuisine: str
    candidates: list[CandidateItem]

# 4. The Prediction Endpoint
@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    # Flatten the incoming JSON into a list of dictionaries
    interactions = []
    for candidate in request.candidates:
        row = request.dict(exclude={'candidates'})
        row.update(candidate.dict())
        interactions.append(row)
        
    # Convert to Pandas DataFrame
    df = pd.DataFrame(interactions)
    
    # Process features using your saved encoders (Simplifying your FeatureEngineer logic)
    for col, le in feature_encoders.items():
        if col in df.columns:
            # 1. Force to string and strip any accidental spaces from the JSON
            df[col] = df[col].astype(str).str.strip()
            
            # 2. The Bulletproof Check: If the label is unknown, use a safe default
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            
            # 3. Now it is 100% safe to transform
            df[col] = le.transform(df[col])
            
    # Convert booleans
    if 'is_weekend' in df.columns:
        df['is_weekend'] = df['is_weekend'].astype(int)
        
    # Ensure column order matches training exactly
    features = ranker_model.feature_names_in_
    X = df[features]
    
    # Predict continuous ranking scores
    # Predict and sort
    scores = ranker_model.predict(X)
    df['predicted_score'] = scores # Let Pandas handle the array!
    
    # Return the original candidate IDs sorted by the model's score
    ranked_candidates = df.sort_values(by='predicted_score', ascending=False)
    
    results = []
    for rank, (_, row) in enumerate(ranked_candidates.iterrows(), 1):
        
        item_id = str(row['candidate_item_id']).strip().lower()
        # Look up the human-readable name, default to 'Unknown' if not found
        dish_name = item_id_to_name.get(item_id, "Unknown Dish")
        
        results.append({
            "rank": rank,
            "item_id": item_id,
            "item_name": dish_name, # <-- The new human-readable name!
            "score": round(float(row['predicted_score']), 4)
        })
        
    return {
        "session_id": request.session_id,
        "user_context": {
            "user_id": request.user_id,
            "persona": request.user_demographic_cluster,
            "current_cart_theme": request.dominant_cuisine,
            "time_of_day": request.time_of_day,
            "weather": request.weather_proxy,
            "cart_value": request.current_cart_value_inr
        },
        "recommendations": results
    }