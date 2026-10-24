# Q-FEATURE: Quantum QUBO + QAOA Robust Feature Selection

Q-FEATURE is a hybrid classical-quantum machine learning pipeline that formulates feature selection as a multi-objective Quadratic Unconstrained Binary Optimization (QUBO) problem solved using QAOA (Quantum Approximate Optimization Algorithm) on Qiskit Aer.

## Features
- **Relevance:** Mutual Information ranking against target attack categories.
- **Redundancy:** Pairwise Spearman/Pearson correlation matrix penalizing collinear features.
- **Robustness/Instability:** Feature degradation under noise perturbation.
- **Quantum Optimization:** Parameterized QAOA ansatz with variational angle optimization.
- **Benchmark Baselines:** Mutual Information, RFE-SVM, XGBoost, and classical exact QUBO.

## Installation
Using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Running the Pipeline
```bash
python main.py
```
