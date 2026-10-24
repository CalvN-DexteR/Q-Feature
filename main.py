import os
import json
from src.data_loader import download_data, load_and_preprocess
from src.feature_scoring import generate_qubo_components, calculate_relevance
from src.qubo_qaoa import formulate_qubo, solve_exact, solve_qaoa
from src.baselines import train_and_evaluate, evaluate_robustness, get_xgboost_importance, get_rfe_svm_features

def run_experiment():
    import numpy as np
    np.random.seed(42)  # Ensure 100% reproducibility for Gaussian perturbations
    
    print("--- Q-FEATURE Experiment Pipeline ---")
    # 1. Load Data
    train_path, test_path = download_data()
    data = load_and_preprocess(train_path, test_path)
    
    n_features = len(data['feature_names'])
    print(f"Total features: {n_features}")
    
    # 2. Get QUBO components
    # Using top 12 features
    m = 12
    k = 6
    
    top_features, r_norm, d_matrix, u_norm = generate_qubo_components(data, m=m, R=5, sigma=1.0)
    print(f"Top {m} features (MI): {top_features}")
    
    # 3. Baselines feature selection
    print("Running baselines feature selection...")
    xgb_features = get_xgboost_importance(data['X_train_full'], data['y_train_full'], k=10)
    print(f"XGBoost top 10: {xgb_features}")
    
    rfe_features = get_rfe_svm_features(data['X_train_full'], data['y_train_full'], k=8)
    print(f"RFE SVM top 8: {rfe_features}")
    
    # 4. QUBO Optimization
    print("Formulating QUBO...")
    qubo = formulate_qubo(r_norm, d_matrix, u_norm, k=k, alpha=1.0, beta=1.0, eta=1.0, A=10.0)
    
    print("Solving Exact Classical QUBO...")
    mask_classical, cost_classical, time_classical = solve_exact(qubo)
    classical_qubo_features = [top_features[i] for i, val in enumerate(mask_classical) if val == 1.0]
    print(f"Classical QUBO features ({len(classical_qubo_features)}): {classical_qubo_features}")
    
    print("Solving QAOA...")
    mask_qaoa, cost_qaoa, time_qaoa, result_qaoa = solve_qaoa(qubo, p=1, shots=1024, maxiter=50)
    qaoa_features = [top_features[i] for i, val in enumerate(mask_qaoa) if val == 1.0]
    print(f"QAOA features ({len(qaoa_features)}): {qaoa_features}")
    
    # Define models to evaluate
    models_to_eval = [
        {"name": "All Features + RF", "features": data['feature_names']},
        {"name": "MI + RF", "features": top_features},
        {"name": "RFE + SVM", "features": rfe_features},
        {"name": "XGBoost + RF", "features": xgb_features},
        {"name": "Classical QUBO + RF", "features": classical_qubo_features},
        {"name": "Q-FEATURE + RF", "features": qaoa_features},
    ]
    
    results = []
    
    for model in models_to_eval:
        print(f"\nEvaluating {model['name']} with {len(model['features'])} features...")
        feat_subset = model['features']
        
        # Train and Evaluate
        X_train_sub = data['X_train_full'][feat_subset]
        X_test_sub = data['X_test'][feat_subset]
        
        rf, acc, prec, rec, f1, fpr, train_time = train_and_evaluate(
            X_train_sub, data['y_train_full'], X_test_sub, data['y_test']
        )
        
        # Robustness
        f1_clean, f1_pert, rob_deg = evaluate_robustness(
            rf, X_test_sub, data['y_test'], feat_subset, R=5, sigma=1.0
        )
        
        feature_red = (1 - len(feat_subset) / n_features) * 100
        
        results.append({
            "Method": model['name'],
            "Features": len(feat_subset),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "FPR": fpr,
            "Clean F1": f1_clean,
            "Perturbed F1": f1_pert,
            "RD (pp)": rob_deg,
            "Feature Reduction (%)": feature_red
        })
        
    # Save results to a json
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Pipeline Complete ---")
    print("Results saved to results.json")
    
    # 5. Visualize Results
    from src.visualization import plot_results
    plot_results('results.json')

if __name__ == "__main__":
    run_experiment()
