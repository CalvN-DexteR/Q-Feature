import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Qiskit 1.0+ / 2.0+ and Qiskit-Aer imports
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator

def formulate_qubo(r, d, u, k=6, alpha=1.0, beta=1.0, eta=1.0, A=10.0):
    m = len(r)
    return {
        'r': r,
        'd': d,
        'u': u,
        'k': k,
        'alpha': alpha,
        'beta': beta,
        'eta': eta,
        'A': A,
        'm': m
    }

def build_qubo_matrix(qubo):
    """
    Constructs the QUBO matrix Q from relevance (r), redundancy (d),
    instability (u), and the cardinality penalty A * (sum(x) - k)^2.
    
    Penalty expansion:
      A * (sum(x) - k)^2 = A * [ sum_i (1 - 2k) x_i + 2 * sum_{i<j} x_i x_j + k^2 ]
    
    Total Cost(x) = x^T Q x + const
    """
    r, d, u, k, m = qubo['r'], qubo['d'], qubo['u'], qubo['k'], qubo['m']
    alpha, beta, eta = qubo['alpha'], qubo['beta'], qubo['eta']
    A = qubo.get('A', 10.0)
    
    Q = np.zeros((m, m))
    for i in range(m):
        # Linear term for x_i
        Q[i, i] = -alpha * r[i] + eta * u[i] + A * (1.0 - 2.0 * k)
        for j in range(i + 1, m):
            # Quadratic cross-term for x_i * x_j
            Q[i, j] = beta * d[i, j] + 2.0 * A
            
    return Q

def qubo_to_ising_operator(Q):
    """
    Converts upper-triangular QUBO matrix Q into an Ising Hamiltonian (SparsePauliOp)
    using the mapping x_i = (I - Z_i) / 2.
    """
    m = Q.shape[0]
    pauli_terms = []
    
    # 1. Linear Pauli-Z terms
    for i in range(m):
        p_str = ['I'] * m
        p_str[m - 1 - i] = 'Z'
        h_i = -0.5 * Q[i, i]
        for j in range(i + 1, m):
            h_i -= 0.25 * Q[i, j]
        for j in range(i):
            h_i -= 0.25 * Q[j, i]
        pauli_terms.append((''.join(p_str), float(h_i)))
        
    # 2. Quadratic Pauli-ZZ terms (i < j)
    for i in range(m):
        for j in range(i + 1, m):
            p_str = ['I'] * m
            p_str[m - 1 - i] = 'Z'
            p_str[m - 1 - j] = 'Z'
            J_ij = 0.25 * Q[i, j]
            pauli_terms.append((''.join(p_str), float(J_ij)))
            
    return SparsePauliOp.from_list(pauli_terms).simplify()

def cost_function(mask_indices, qubo):
    r, d, u = qubo['r'], qubo['d'], qubo['u']
    alpha, beta, eta = qubo['alpha'], qubo['beta'], qubo['eta']
    
    cost = 0.0
    for i in mask_indices:
        cost += -alpha * r[i] + eta * u[i]
        
    for i in mask_indices:
        for j in mask_indices:
            if i < j:
                cost += beta * d[i, j]
                
    return cost

def solve_exact(qubo):
    m = qubo['m']
    k = qubo['k']
    
    start_time = time.time()
    best_cost = float('inf')
    best_mask = None
    
    # Enumerate all C(m, k) combinations
    for combo in combinations(range(m), k):
        cost = cost_function(combo, qubo)
        if cost < best_cost:
            best_cost = cost
            best_mask = combo
            
    runtime = time.time() - start_time
    
    mask = np.zeros(m)
    for idx in best_mask:
        mask[idx] = 1.0
        
    return mask, best_cost, runtime

def solve_qaoa(qubo, p=1, shots=2048, maxiter=50, random_state=42):
    """
    Executes real QAOA simulation using Qiskit Aer:
      1. Converts QUBO to an Ising Cost Hamiltonian.
      2. Constructs parameterized QAOA circuit (QAOAAnsatz) with p layers.
      3. Uses Qiskit Aer Estimator (approximation=True) for exact, shot-noise-free
         expectation value evaluation during classical COBYLA optimization.
      4. Samples the final quantum state using AerSimulator with shots.
      5. Extracts valid bitstrings satisfying sum(x) == k and computes feasibility telemetry.
    """
    start_time = time.time()
    m = qubo['m']
    k = qubo['k']
    
    # 1. Build QUBO matrix and Ising operator
    Q = build_qubo_matrix(qubo)
    cost_op = qubo_to_ising_operator(Q)
    
    # 2. Build QAOA Ansatz circuit
    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=p)
    
    # 3. Exact expectation evaluation for smooth, deterministic COBYLA optimization
    estimator = Estimator(approximation=True)
    
    def objective_fn(params):
        res = estimator.run([ansatz], [cost_op], [params]).result()
        return res.values[0]

    # 4. Variational parameter initialization (Trotterized Quantum Annealing schedule)
    gamma_init = np.linspace(0.1, 0.5, p)
    beta_init = np.linspace(0.5, 0.1, p)
    init_params = np.concatenate([gamma_init, beta_init])
    
    opt_result = minimize(objective_fn, init_params, method='COBYLA', options={'maxiter': maxiter})
    
    # 5. Final measurement sampling on AerSimulator with optimal variational angles
    circuit_for_sampling = ansatz.copy()
    circuit_for_sampling.measure_all()
    backend = AerSimulator(method='statevector', seed_simulator=random_state)
    transpiled_circuit = transpile(circuit_for_sampling, backend)
    optimal_circ = transpiled_circuit.assign_parameters(opt_result.x)
    
    final_job = backend.run(optimal_circ, shots=max(shots, 2048))
    counts = final_job.result().get_counts()
    
    # 6. Extract and filter bitstrings that satisfy cardinality constraint sum(x) == k
    valid_candidates = []
    total_valid_shots = 0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Little-endian bitstring mapping
        x = np.array([int(b) for b in bitstring[::-1]])
        if np.sum(x) == k:
            total_valid_shots += count
            indices = [i for i, val in enumerate(x) if val == 1]
            cost = cost_function(indices, qubo)
            valid_candidates.append((x, cost, count))
            
    feasibility_ratio = total_valid_shots / total_shots if total_shots > 0 else 0.0
    fallback_used = False
    
    if valid_candidates:
        # Sort by minimum cost, breaking ties by frequency/count
        valid_candidates.sort(key=lambda item: (item[1], -item[2]))
        best_mask, best_cost, best_count = valid_candidates[0]
        print(f"  [QAOA] Feasibility Ratio: {feasibility_ratio * 100:.2f}% ({total_valid_shots}/{total_shots} shots). "
              f"Best cost: {best_cost:.4f} (sampled {best_count} times).")
    else:
        print(f"  [WARNING] QAOA failed to sample any valid bitstrings satisfying sum(x) == {k} "
              f"(feasibility: 0.00%). Falling back to classical exact ground state.")
        fallback_used = True
        best_mask, best_cost, _ = solve_exact(qubo)
        
    runtime = time.time() - start_time
    diagnostics = {
        'feasibility_ratio': feasibility_ratio,
        'fallback_used': fallback_used,
        'valid_candidates_count': len(valid_candidates),
        'optimal_params': opt_result.x.tolist(),
        'optimization_energy': float(opt_result.fun)
    }
    
    return best_mask, best_cost, runtime, opt_result, diagnostics
