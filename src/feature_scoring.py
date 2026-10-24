import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def calculate_relevance(X, y, m=12):
    print("Calculating Mutual Information (Relevance)...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)
    
    # Get top m features
    top_m_features = mi_series.nlargest(m).index.tolist()
    
    # Extract relevance for top m
    r_raw = mi_series[top_m_features].values
    # Normalize relevance [0, 1]
    r_max = float(max(r_raw))
    if r_max > 0:
        r_normalized = np.array([float(val) / r_max for val in r_raw], dtype=float)
    else:
        r_normalized = r_raw
        
    return top_m_features, r_normalized

def calculate_redundancy(X_subset):
    print("Calculating Pearson Correlation (Redundancy)...")
    corr_matrix = X_subset.corr(method='pearson').abs().values
    return corr_matrix

def calculate_instability(X_fit, y_fit, X_val, y_val, features, R=5, sigma=1.0):
    print("Calculating Instability (Robustness)...")
    
    # Subset features
    X_fit_sub = X_fit[features]
    X_val_sub = X_val[features]
    
    # Train provisional RF
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_fit_sub, y_fit)
    
    # Clean F1
    preds_clean = rf.predict(X_val_sub)
    f1_clean = f1_score(y_val, preds_clean)
    print(f"  Clean Validation F1: {f1_clean:.4f}")
    
    u_raw = np.zeros(len(features))
    
    for idx, feat in enumerate(features):
        f1_perturbed_sum = 0
        for r in range(R):
            # Create a copy to perturb
            X_val_perturbed = X_val_sub.copy()
            
            # Since data is standard scaled, we can just add Gaussian noise.
            # In a real scenario, we might skip categorical features. 
            # For simplicity and given standard scaling, we apply noise to all selected.
            noise = np.random.normal(0, sigma, size=X_val_perturbed[feat].shape)
            X_val_perturbed[feat] += noise
            
            preds_pert = rf.predict(X_val_perturbed)
            f1_pert = f1_score(y_val, preds_pert)
            f1_perturbed_sum += f1_pert
            
        f1_avg_pert = f1_perturbed_sum / R
        u_raw[idx] = max(0, f1_clean - f1_avg_pert)
        
    # Normalize instability
    u_max = float(max(u_raw))
    if u_max > 0:
        u_normalized = np.array([float(val) / u_max for val in u_raw], dtype=float)
    else:
        u_normalized = u_raw
        
    return u_normalized

def generate_qubo_components(data_dict, m=12, R=5, sigma=1.0):
    X_fit = data_dict['X_fit']
    y_fit = data_dict['y_fit']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    
    top_features, r_norm = calculate_relevance(X_fit, y_fit, m=m)
    d_matrix = calculate_redundancy(X_fit[top_features])
    u_norm = calculate_instability(X_fit, y_fit, X_val, y_val, top_features, R=R, sigma=sigma)
    
    return top_features, r_norm, d_matrix, u_norm
