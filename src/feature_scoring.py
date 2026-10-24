import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def calculate_relevance(X, y, m=12, random_state=42):
    print("Calculating Mutual Information (Relevance)...")
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=X.columns)
    
    # Get top m features
    top_m_features = mi_series.nlargest(m).index.tolist()
    
    # Extract relevance for top m
    r_raw = mi_series[top_m_features].values
    # Normalize relevance [0, 1]
    if r_raw.max() > 0:
        r_normalized = r_raw / r_raw.max()
    else:
        r_normalized = r_raw
        
    return top_m_features, r_normalized

def calculate_redundancy(X_subset, method='spearman'):
    print(f"Calculating Correlation Matrix (Redundancy, method={method})...")
    # Spearman rank correlation detects non-linear monotonic relationships; Pearson detects linear
    corr_matrix = X_subset.corr(method=method).abs().values
    # Zero out diagonal self-correlation
    np.fill_diagonal(corr_matrix, 0.0)
    return corr_matrix

def calculate_instability(X_fit, y_fit, X_val, y_val, features, categorical_cols=None, R=5, sigma=1.0, random_state=42):
    print("Calculating Instability (Robustness)...")
    rng = np.random.default_rng(random_state)
    categorical_set = set(categorical_cols or [])
    
    # Subset features
    X_fit_sub = X_fit[features]
    X_val_sub = X_val[features]
    
    # Train provisional RF
    rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    rf.fit(X_fit_sub, y_fit)
    
    # Clean F1
    preds_clean = rf.predict(X_val_sub)
    f1_clean = f1_score(y_val, preds_clean)
    print(f"  Clean Validation F1: {f1_clean:.4f}")
    
    u_raw = np.zeros(len(features))
    
    for idx, feat in enumerate(features):
        f1_perturbed_sum = 0.0
        is_categorical = feat in categorical_set
        
        for r in range(R):
            X_val_perturbed = X_val_sub.copy()
            
            if is_categorical:
                # For categorical features: random permutation/replacement from observed column distribution
                col_vals = X_val_sub[feat].to_numpy()
                shuffled = rng.choice(col_vals, size=len(col_vals), replace=True)
                X_val_perturbed[feat] = shuffled
            else:
                # For continuous features: additive Gaussian noise
                noise = rng.normal(0, sigma, size=len(X_val_perturbed))
                X_val_perturbed[feat] = X_val_perturbed[feat] + noise
            
            preds_pert = rf.predict(X_val_perturbed)
            f1_pert = f1_score(y_val, preds_pert)
            f1_perturbed_sum += f1_pert
            
        f1_avg_pert = f1_perturbed_sum / R
        u_raw[idx] = max(0.0, f1_clean - f1_avg_pert)
        
    # Robust normalization: avoid inflating negligible noise drops (< 0.05) to 1.0
    if u_raw.max() > 1e-4:
        norm_scale = max(u_raw.max(), 0.05)
        u_normalized = u_raw / norm_scale
    else:
        u_normalized = u_raw
        
    return u_normalized

def generate_qubo_components(data_dict, m=12, R=5, sigma=1.0, redundancy_method='spearman', random_state=42):
    X_fit = data_dict['X_fit']
    y_fit = data_dict['y_fit']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    categorical_cols = data_dict.get('categorical_cols', [])
    
    top_features, r_norm = calculate_relevance(X_fit, y_fit, m=m, random_state=random_state)
    d_matrix = calculate_redundancy(X_fit[top_features], method=redundancy_method)
    u_norm = calculate_instability(
        X_fit, y_fit, X_val, y_val, top_features,
        categorical_cols=categorical_cols, R=R, sigma=sigma, random_state=random_state
    )
    
    return top_features, r_norm, d_matrix, u_norm
