import os
import json
import argparse
import numpy as np
from src.data_loader import download_data, load_and_preprocess
from src.feature_scoring import generate_qubo_components
from src.qubo_qaoa import formulate_qubo, solve_exact, solve_qaoa
from src.baselines import train_and_evaluate, evaluate_robustness, get_xgboost_importance, get_rfe_svm_features
from src.visualization import plot_results

def parse_args():
    parser = argparse.ArgumentParser(description="Q-FEATURE: Quantum QUBO + QAOA Robust Feature Selection")
    parser.add_argument("--k", type=int, default=6, help="Target number of features to select (default: 6)")
    parser.add_argument("--m", type=int, default=12, help="Number of candidate features from MI pre-filter (default: 12)")
    parser.add_argument("--sample_size", type=int, default=10000, help="Subsample size for train/test sets (0 for full dataset, default: 10000)")
    parser.add_argument("--p", type=int, default=1, help="Number of QAOA ansatz layers (default: 1)")
    parser.add_argument("--shots", type=int, default=2048, help="Measurement shots for QAOA sampling (default: 2048)")
    parser.add_argument("--maxiter", type=int, default=50, help="Optimizer max iterations for QAOA (default: 50)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Relevance weight (default: 1.0)")
    parser.add_argument("--beta", type=float, default=1.0, help="Redundancy weight (default: 1.0)")
    parser.add_argument("--eta", type=float, default=1.0, help="Instability weight (default: 1.0)")
    parser.add_argument("--A", type=float, default=10.0, help="Cardinality penalty coefficient A (default: 10.0)")
    parser.add_argument("--redundancy_method", type=str, default="spearman", choices=["spearman", "pearson"], help="Redundancy metric (default: spearman)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()

def run_experiment(args=None):
    if args is None:
        args = parse_args()
        
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("               Q-FEATURE EXPERIMENT PIPELINE")
    print(f"  Configuration: k={args.k}, m={args.m}, p={args.p}, sample_size={args.sample_size}, redundancy={args.redundancy_method}")
    print("=" * 80)
    
    # 1. Load Data
    train_path, test_path = download_data()
    sample_sz = args.sample_size if args.sample_size > 0 else None
    data = load_and_preprocess(train_path, test_path, sample_size=sample_sz, random_state=args.seed)
    
    n_features = len(data['feature_names'])
    print(f"Total available features: {n_features}")
    print(f"Continuous features ({len(data['continuous_cols'])}): {data['continuous_cols']}")
    print(f"Categorical features ({len(data['categorical_cols'])}): {data['categorical_cols']}")
    
    # 2. Get QUBO components
    m = args.m
    k = args.k
    
    top_candidates, r_norm, d_matrix, u_norm = generate_qubo_components(
        data, m=m, R=5, sigma=1.0, redundancy_method=args.redundancy_method, random_state=args.seed
    )
    print(f"\nTop {m} MI Candidates: {top_candidates}")
    
    # 3. Baselines feature selection (Fair comparison: all evaluated at identical feature budget k)
    print("\nRunning baseline feature selectors (all standardized to k={k})...".format(k=k))
    xgb_features = get_xgboost_importance(data['X_train_full'], data['y_train_full'], k=k, random_state=args.seed)
    print(f"  • XGBoost (top {k}): {xgb_features}")
    
    rfe_features = get_rfe_svm_features(data['X_train_full'], data['y_train_full'], k=k, random_state=args.seed)
    print(f"  • RFE SVM (top {k}): {rfe_features}")
    
    mi_top_k = top_candidates[:k]
    print(f"  • MI (top {k}): {mi_top_k}")
    
    # 4. QUBO Optimization
    print("\nFormulating QUBO...")
    qubo = formulate_qubo(r_norm, d_matrix, u_norm, k=k, alpha=args.alpha, beta=args.beta, eta=args.eta, A=args.A)
    
    print("Solving Exact Classical QUBO (exhaustive ground state search)...")
    mask_classical, cost_classical, time_classical = solve_exact(qubo)
    classical_qubo_features = [top_candidates[i] for i, val in enumerate(mask_classical) if val == 1.0]
    print(f"  • Classical QUBO features ({len(classical_qubo_features)}): {classical_qubo_features} (Cost: {cost_classical:.4f}, Time: {time_classical:.4f}s)")
    
    print("\nSolving Quantum QAOA (Qiskit Aer Simulator)...")
    mask_qaoa, cost_qaoa, time_qaoa, opt_qaoa, qaoa_diag = solve_qaoa(
        qubo, p=args.p, shots=args.shots, maxiter=args.maxiter, random_state=args.seed
    )
    qaoa_features = [top_candidates[i] for i, val in enumerate(mask_qaoa) if val == 1.0]
    print(f"  • QAOA features ({len(qaoa_features)}): {qaoa_features} (Cost: {cost_qaoa:.4f}, Time: {time_qaoa:.4f}s)")
    print(f"  • QAOA Feasibility: {qaoa_diag['feasibility_ratio']*100:.2f}%, Fallback used: {qaoa_diag['fallback_used']}")
    
    # 5. Define models to evaluate
    models_to_eval = [
        {"name": "All Features + RF", "features": data['feature_names']},
        {"name": f"MI (top {k}) + RF", "features": mi_top_k},
        {"name": f"RFE SVM (top {k}) + RF", "features": rfe_features},
        {"name": f"XGBoost (top {k}) + RF", "features": xgb_features},
        {"name": f"Classical QUBO (k={k}) + RF", "features": classical_qubo_features},
        {"name": f"Q-FEATURE (QAOA k={k}) + RF", "features": qaoa_features},
    ]
    
    results = []
    
    print("\n" + "=" * 80)
    print("                 MODEL TRAINING & EVALUATION")
    print("=" * 80)
    
    for model in models_to_eval:
        feat_subset = model['features']
        print(f"\nEvaluating {model['name']} with {len(feat_subset)} features...")
        
        # Train and Evaluate
        X_train_sub = data['X_train_full'][feat_subset]
        X_test_sub = data['X_test'][feat_subset]
        
        rf, acc, prec, rec, f1, fpr, train_time = train_and_evaluate(
            X_train_sub, data['y_train_full'], X_test_sub, data['y_test'], random_state=args.seed
        )
        
        # Robustness with domain-aware perturbation
        f1_clean, f1_pert, rob_deg = evaluate_robustness(
            rf, X_test_sub, data['y_test'], feat_subset,
            categorical_cols=data['categorical_cols'], R=5, sigma=1.0, random_state=args.seed
        )
        
        feature_red = (1 - len(feat_subset) / n_features) * 100
        
        results.append({
            "Method": model['name'],
            "Features": len(feat_subset),
            "Selected_Features": feat_subset,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "FPR": fpr,
            "Clean F1": f1_clean,
            "Perturbed F1": f1_pert,
            "RD (pp)": rob_deg,
            "Feature Reduction (%)": feature_red,
            "Train Time (s)": train_time
        })
        
    # Append QAOA telemetry to output
    pipeline_output = {
        "config": vars(args),
        "qaoa_diagnostics": qaoa_diag,
        "results": results
    }
    
    # Save results to json (flat list for backward compatibility with plotting, with metadata attached)
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    with open('experiment_metadata.json', 'w') as f:
        json.dump(pipeline_output, f, indent=4)
        
    print("\n--- Pipeline Complete ---")
    print("Results saved to results.json and experiment_metadata.json")
    
    # 6. Visualize Results
    plot_results('results.json')

if __name__ == "__main__":
    run_experiment()
