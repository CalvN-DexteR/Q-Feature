import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    preds = rf.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return rf, acc, prec, rec, f1, fpr, train_time

def evaluate_robustness(rf, X_test, y_test, features, R=5, sigma=1.0):
    preds_clean = rf.predict(X_test)
    f1_clean = f1_score(y_test, preds_clean)
    
    f1_perturbed_sum = 0
    
    for r in range(R):
        X_test_perturbed = X_test.copy()
        
        # Perturb all used features
        for feat in features:
            noise = np.random.normal(0, sigma, size=X_test_perturbed[feat].shape)
            X_test_perturbed[feat] += noise
            
        preds_pert = rf.predict(X_test_perturbed)
        f1_pert = f1_score(y_test, preds_pert)
        f1_perturbed_sum += f1_pert
        
    f1_avg_pert = f1_perturbed_sum / R
    robustness_degradation = max(0, f1_clean - f1_avg_pert)
    
    return f1_clean, f1_avg_pert, robustness_degradation

def get_xgboost_importance(X_train, y_train, k=10):
    xgb = XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    importances = list(xgb.feature_importances_)
    top_indices = sorted(range(len(importances)), key=lambda i: float(importances[i]), reverse=True)[:k]
    top_features = [X_train.columns[i] for i in top_indices]
    return top_features

def get_rfe_svm_features(X_train, y_train, k=8):
    # Using LinearSVC for speed on large datasets
    svc = LinearSVC(random_state=42, max_iter=1000)
    rfe = RFE(estimator=svc, n_features_to_select=k, step=5)
    rfe.fit(X_train, y_train)
    
    top_features = X_train.columns[rfe.support_].tolist()
    return top_features
