from xml.parsers.expat import model

import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.model_selection import train_test_split
from src.features.build_features import FeatureEngineer
from sklearn.metrics import classification_report

def train_model():
    os.makedirs('src\models\checkpoints', exist_ok=True)

    
    file_path = 'data/processed/synthetic_interactions_10k.csv'
    print(f"Loading dataset from: {os.path.abspath(file_path)}")
    df = pd.read_csv(file_path)
    
    print(f"TOTAL UNIQUE SESSIONS IN FILE: {df['session_id'].nunique()}")
    
    df.columns = df.columns.str.strip()
    

    # 1. Sort the dataframe by session_id
    df = df.sort_values(by='session_id').reset_index(drop=True)
    
    # 2. Extract Features and Target
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    
    # Extract the session_ids to use as "Query IDs" (qid)
    
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

    # Filter out sessions where the user didn't accept anything
    valid_sessions = df_test.groupby('session_id').filter(lambda x: x['actual_accepted'].sum() > 0)
    

    hits = valid_sessions.groupby('session_id').apply(check_hit_at_k, include_groups=False, k=3)
    hit_rate = hits.mean() * 100
    
    print(f"True Hit Rate @ 3: {hit_rate:.2f}%")
    print(f"(Out of {len(hits)} valid test sessions with purchases, the correct item was in the Top 3 for {hits.sum()} of them.)")


    # --- FORMAL OFFLINE EVALUATION METRICS ---
    from sklearn.metrics import roc_auc_score, ndcg_score
    import numpy as np

    print("\n--- Formal Offline Metrics ---")
    
    # 1. Global AUC (Area Under the ROC Curve)
    auc = roc_auc_score(df_test['actual_accepted'], df_test['predicted_score'])
    print(f"Global AUC: {auc:.4f}")

    # Variables for K-based metrics
    K = 3
    precisions = []
    recalls = []
    ndcgs = []

    # Using the same valid_sessions (sessions with at least one '1') from the Hit Rate code
    for session_id, group in valid_sessions.groupby('session_id'):
        # Sort items by highest predicted score
        sorted_group = group.sort_values(by='predicted_score', ascending=False)
        actuals = sorted_group['actual_accepted'].values
        
        # True number of relevant items in this session
        total_relevant = actuals.sum()
        
        # The actual labels of the Top K items the model chose
        top_k_actuals = actuals[:K]
        
        # Precision@K: (Relevant items in Top K) / K
        precisions.append(top_k_actuals.sum() / K)
        
        # Recall@K: (Relevant items in Top K) / (Total Relevant)
        recalls.append(top_k_actuals.sum() / total_relevant)
        
        # NDCG@K (Requires 2D arrays for sklearn)
        y_true_session = np.asarray([group['actual_accepted'].values])
        y_score_session = np.asarray([group['predicted_score'].values])
        
        if len(actuals) > 1: # Safety check to ensure there are items to rank
            session_ndcg = ndcg_score(y_true_session, y_score_session, k=K)
            ndcgs.append(session_ndcg)

    print(f"Precision@{K}: {np.mean(precisions):.4f}")
    print(f"Recall@{K}:    {np.mean(recalls):.4f}")
    print(f"NDCG@{K}:      {np.mean(ndcgs):.4f}")



    # Save Model & Encoders
    joblib.dump(model, 'src/models/checkpoints/ranker.pkl')
    fe.save_encoders()
    model.save_model('src/models/checkpoints/xgb_ranker.json')

if __name__ == "__main__":
    train_model()
