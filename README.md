# Context-Aware Food Recommendation Engine 🍔📈

A production-grade, context-aware recommendation system built with **XGBoost (XGBRanker)** and served via a **FastAPI** REST backend. 

Unlike traditional item-to-item collaborative filtering (which only looks at "users who bought X also bought Y"), this engine utilizes **Pairwise Learning-to-Rank**. It evaluates a user's entire real-time context—including their persona, time of day, weather, and current cart theme—to dynamically sort and recommend the most relevant menu items.

## 🧠 System Architecture

* **Model:** `XGBRanker` with `rank:pairwise` objective.
* **API Framework:** FastAPI for high-performance, asynchronous JSON serving.
* **Data Pipeline:** Custom feature engineering with safe categorical encoding (handling unseen runtime labels gracefully).
* **Evaluation:** Strict offline metrics ignoring "empty sessions" to reflect true real-world conversion rates.

## 📂 Project Structure

```text
zomathon-csao-recsys/
│
├── data/
│   ├── raw/
│   │   └── pfc_menu.csv                 # Raw menu database
│   └── processed/
│       └── synthetic_interactions_10k.csv # Generated training sessions
│
├── src/
│   └── models/
│       ├── train_ranker.py              # Training pipeline & metric calculation
│       └── checkpoints/
│           ├── xgb_ranker.json          # Compiled XGBoost model
│           └── feature_encoders.pkl     # Saved LabelEncoders
│
├── app.py                               # FastAPI backend server
├── test_api.py                          # Client script to test the live API
└── README.md