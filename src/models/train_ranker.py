import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.model_selection import train_test_split
from src.features.build_features import FeatureEngineer
from sklearn.metrics import classification_report

def train_model():
    os.makedirs('src\models\checkpoints', exist_ok=True)

    # --- ADD THESE DEBUG LINES ---
    file_path = 'data/processed/synthetic_interactions_10k.csv'
    print(f"Loading dataset from: {os.path.abspath(file_path)}")
    df = pd.read_csv(file_path)
    
    print(f"TOTAL UNIQUE SESSIONS IN FILE: {df['session_id'].nunique()}")
    
    df.columns = df.columns.str.strip()
    

    # 1. Sort the dataframe by session_id (Strict requirement for XGBRanker)
    df = df.sort_values(by='session_id').reset_index(drop=True)
    
    # 2. Extract Features and Target
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    
    # Extract the session_ids to use as our "Query IDs" (qid)
    # --- ADD THIS LINE TO CONVERT STRINGS TO INTEGERS ---
    df['session_id_int'] = pd.factorize(df['session_id'])[0]
    session_ids = df['session_id_int']
    # ----------------------------------------------------
    
    # 3. Temporal Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    qid_train, qid_test = session_ids.iloc[:split_idx], session_ids.iloc[split_idx:]
    
    print(f"Training XGBoost Ranker on {len(qid_train.unique())} sessions, testing on {len(qid_test.unique())}...")
    
    # 4. Initialize the True Ranker
    model = xgb.XGBRanker(
        objective='rank:pairwise', # Learns to push the '1' above the '0's
        eval_metric='ndcg',        # Standard ranking metric
        learning_rate=0.05,
        max_depth=5,
        n_estimators=100
    )
    
    # 5. Fit using the qid (Query ID) parameter
    model.fit(
        X_train, y_train, 
        qid=qid_train,
        eval_set=[(X_test, y_test)], 
        eval_qid=[qid_test],
        verbose=10 
    )
    
    print(f"\nRanker Trained Successfully!")
    print(f"Accuracy (Training): {model.score(X_train, y_train):.4f}")
    print(f"Accuracy (Testing): {model.score(X_test, y_test):.4f}")

    print("\n--- Baseline Distribution (Test Set) ---")
    # This shows what percentage of test data is 0s vs 1s
    print((y_test.value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    
    

    # --- RANKING EVALUATION BLOCK ---
    print("\n--- True Ranking Evaluation ---")
    
    y_scores = model.predict(X_test)
    df_test = df.iloc[split_idx:].copy()
    df_test['predicted_score'] = y_scores
    df_test['actual_accepted'] = y_test.values
    df_test = df_test.sort_values(by=['session_id', 'predicted_score'], ascending=[True, False])
    
    def check_hit_at_k(group, k=3):
        top_k = group.head(k)
        return top_k['actual_accepted'].sum() > 0

    # 1. NEW: Filter out sessions where the user didn't accept anything
    valid_sessions = df_test.groupby('session_id').filter(lambda x: x['actual_accepted'].sum() > 0)
    
    # 2. UPDATED: Calculate hits only on valid sessions
    hits = valid_sessions.groupby('session_id').apply(check_hit_at_k, include_groups=False, k=3)
    hit_rate = hits.mean() * 100
    
    print(f"True Hit Rate @ 3: {hit_rate:.2f}%")
    print(f"(Out of {len(hits)} valid test sessions with purchases, the correct item was in the Top 3 for {hits.sum()} of them.)")

    # Save Model & Encoders
    joblib.dump(model, 'src/models/checkpoints/ranker.pkl')
    fe.save_encoders()

if __name__ == "__main__":
    train_model()