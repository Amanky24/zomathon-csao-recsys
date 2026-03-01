import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.model_selection import train_test_split
from src.features.build_features import FeatureEngineer
from sklearn.metrics import classification_report

def train_model():
    os.makedirs('src\models\checkpoints', exist_ok=True)
    print("Loading 10k interactions dataset...")
    
    
    df = pd.read_csv('src\models\synthetic_interactions_10k_test.csv')
    
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    
    # Temporal Split: shuffle=False ensures we train on the past and test on the future
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training XGBoost Ranker on {len(X_train)} sessions, testing on {len(X_test)}...")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.05,
        max_depth=5,
        n_estimators=100,
        scale_pos_weight=2.25
    )
    
    # Passing eval_set to monitor the AUC on both the training and temporal test sets
    model.fit(
        X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)], 
        verbose=10  # Prints out progress every 10 trees
    )
    
    print(f"\nRanker Trained Successfully!")
    print(f"Accuracy (Training): {model.score(X_train, y_train):.4f}")
    print(f"Accuracy (Testing): {model.score(X_test, y_test):.4f}")

    print("\n--- Baseline Distribution (Test Set) ---")
    # This shows what percentage of test data is 0s vs 1s
    print((y_test.value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    
    print("\n--- Classification Report ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


    
    # --- RANKING EVALUATION BLOCK ---
    print("\n--- Ranking Evaluation ---")
    
    # 1. Get the raw probabilities (confidence scores) for Class 1
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 2. Grab the corresponding rows from the original dataframe
    # Since we didn't shuffle, the test set is exactly the last len(X_test) rows
    test_start_idx = len(df) - len(X_test)
    df_test = df.iloc[test_start_idx:].copy()
    
    # 3. Attach our model's predictions to the test data
    df_test['predicted_prob'] = y_probs
    df_test['actual_accepted'] = y_test.values
    
    # 4. Sort the dataframe by Session ID, and then by our Model's Probability (Highest to Lowest)
    df_test = df_test.sort_values(by=['session_id', 'predicted_prob'], ascending=[True, False])
    
    # 5. Calculate Hit Rate @ 3
    def check_hit_at_k(group, k=3):
        # Take the top K highest-probability items for this session
        top_k = group.head(k)
        # Check if the actually accepted item (1) is in those top K slots
        return top_k['actual_accepted'].sum() > 0

    # Apply the function to each unique session
    hits = df_test.groupby('session_id').apply(check_hit_at_k, k=3)
    hit_rate = hits.mean() * 100
    
    print(f"Hit Rate @ 3: {hit_rate:.2f}%")
    print(f"(Out of {len(hits)} total test sessions, the correct item was in the Top 3 for {hits.sum()} of them.)")
    # --------------------------------


    # Save Model & Encoders
    joblib.dump(model, 'src/models/checkpoints/ranker.pkl')
    fe.save_encoders()

if __name__ == "__main__":
    train_model()