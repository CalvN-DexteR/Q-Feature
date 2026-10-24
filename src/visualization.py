import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_results(json_path='results.json'):
    # Load results
    with open(json_path, 'r') as f:
        results = json.load(f)
        
    df = pd.DataFrame(results)
    
    # Set up the plot style
    plt.style.use('ggplot')
    
    # 1. Plot Performance Metrics (F1, Accuracy, Precision, Recall)
    metrics_to_plot = [m for m in ['Accuracy', 'Precision', 'Recall', 'F1'] if m in df.columns]
    ax = df.plot(x='Method', y=metrics_to_plot, kind='bar', figsize=(12, 6), width=0.8)
    plt.title('Classification Performance Comparison by Method')
    plt.ylabel('Score')
    plt.xlabel('Feature Selection Method')
    plt.ylim(0.0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Saved performance_comparison.png")
    
    # 2. Plot Robustness (Clean vs Perturbed F1)
    robustness_metrics = [m for m in ['Clean F1', 'Perturbed F1'] if m in df.columns]
    ax2 = df.plot(x='Method', y=robustness_metrics, kind='bar', figsize=(10, 6), color=['#1f77b4', '#d62728'])
    plt.title('Robustness Comparison: Clean vs Perturbed F1-Score')
    plt.ylabel('F1-Score')
    plt.xlabel('Feature Selection Method')
    plt.ylim(0.0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('robustness_comparison.png')
    print("Saved robustness_comparison.png")
    
    # 3. Print Tabular Results nicely to console
    print("\n" + "="*95)
    print("FINAL BENCHMARK RESULTS")
    print("="*95)
    
    # Formatting for terminal output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    
    # Candidate columns to display
    desired_cols = ['Method', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1', 'Clean F1', 'Perturbed F1', 'RD (pp)', 'Feature Reduction (%)']
    display_cols = [c for c in desired_cols if c in df.columns]
    df_display = df[display_cols].copy()
    
    # Format numbers for better readability
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'Clean F1', 'Perturbed F1', 'RD (pp)']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
    if 'Feature Reduction (%)' in df_display.columns:
        df_display['Feature Reduction (%)'] = df_display['Feature Reduction (%)'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x))
    
    print(df_display.to_string(index=False))
    print("="*95)
    
    # If selected features are recorded, print summary breakdown
    if 'Selected_Features' in df.columns:
        print("\nSELECTED FEATURE SUBSETS:")
        for idx, row in df.iterrows():
            feat_list = row.get('Selected_Features', [])
            feat_str = ", ".join(feat_list) if isinstance(feat_list, list) else str(feat_list)
            print(f"  • {row['Method']:<22} ({len(feat_list)} feats): {feat_str}")
        print("="*95 + "\n")

if __name__ == "__main__":
    plot_results()
