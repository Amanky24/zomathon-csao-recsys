import requests
import json
import uuid

# The local address where API is listening
API_URL = "http://127.0.0.1:8000/recommend"

# This is the exact JSON structure API is expecting
# Imagine a "Discount Chaser" is ordering Late Night food on a rainy weekend
payload = {
    "session_id": str(uuid.uuid4()),
    "user_id": "U_8492",
    "user_demographic_cluster": "The Discount Chaser",
    "user_segment": "Budget/Student",
    "time_of_day": "Late_Night",
    "is_weekend": True,
    "weather_proxy": "Rainy/Cold",
    "transit_distance_km": 3.5,
    "restaurant_busyness": 8,
    "current_cart_value_inr": 250.0,
    "dominant_cuisine": "Fast Food",
    "candidates": [
        {"candidate_item_id": "pfc_012", "candidate_macro_cat": "Dessert", "candidate_price_ratio": 0.3},
        {"candidate_item_id": "pfc_045", "candidate_macro_cat": "Beverage", "candidate_price_ratio": 0.15},
        {"candidate_item_id": "pfc_003", "candidate_macro_cat": "Main_Course", "candidate_price_ratio": 0.8},
        {"candidate_item_id": "pfc_022", "candidate_macro_cat": "Starter", "candidate_price_ratio": 0.5},
        {"candidate_item_id": "pfc_031", "candidate_macro_cat": "Accompaniment", "candidate_price_ratio": 0.2}
    ]
}



print("Sending Request to Recommendation API...")

# Send the POST request
response = requests.post(API_URL, json=payload)

# Print the results
if response.status_code == 200:
    print("\n✅ Success! Here are the ranked recommendations:\n")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"❌ Error {response.status_code}: {response.text}")