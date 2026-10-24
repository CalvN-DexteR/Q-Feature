import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    preds = rf.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return rf, acc, prec, rec, f1, fpr, train_time

def evaluate_robustness(rf, X_test, y_test, features, categorical_cols=None, R=5, sigma=1.0, random_state=42):
    rng = np.random.default_rng(random_state)
    categorical_set = set(categorical_cols or [])
    
    preds_clean = rf.predict(X_test)
    f1_clean = f1_score(y_test, preds_clean, zero_division=0)
    
    f1_perturbed_sum = 0.0
    
    for r in range(R):
        X_test_perturbed = X_test.copy()
        
        # Perturb all features according to their domain data type
        for feat in features:
            if feat in categorical_set:
                # Categorical column: random category resampling from observed distribution
                col_vals = X_test[feat].to_numpy()
                shuffled = rng.choice(col_vals, size=len(col_vals), replace=True)
                X_test_perturbed[feat] = shuffled
            else:
                # Continuous column: additive Gaussian noise
                noise = rng.normal(0, sigma, size=len(X_test_perturbed))
                X_test_perturbed[feat] = X_test_perturbed[feat] + noise
            
        preds_pert = rf.predict(X_test_perturbed)
        f1_pert = f1_score(y_test, preds_pert, zero_division=0)
        f1_perturbed_sum += f1_pert
        
    f1_avg_pert = f1_perturbed_sum / R
    robustness_degradation = max(0.0, f1_clean - f1_avg_pert)
    
    return f1_clean, f1_avg_pert, robustness_degradation

def get_xgboost_importance(X_train, y_train, k=6, random_state=42):
    xgb = XGBClassifier(n_estimators=50, random_state=random_state, n_jobs=-1, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    importances = xgb.feature_importances_
    
    top_indices = np.argsort(importances)[::-1][:k]
    top_features = X_train.columns[top_indices].tolist()
    return top_features

def get_rfe_svm_features(X_train, y_train, k=6, random_state=42):
    # LinearSVC for fast linear boundary feature elimination
    svc = LinearSVC(random_state=random_state, max_iter=1000, dual='auto')
    rfe = RFE(estimator=svc, n_features_to_select=k, step=5)
    rfe.fit(X_train, y_train)
    
    top_features = X_train.columns[rfe.support_].tolist()
    return top_features
