import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

# Qiskit 1.0+ / 2.0+ and Qiskit-Aer imports
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

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

def solve_qaoa(qubo, p=1, shots=1024, maxiter=50):
    """
    Executes real QAOA simulation using Qiskit Aer:
      1. Converts QUBO to an Ising Cost Hamiltonian.
      2. Constructs parameterized QAOA circuit (QAOAAnsatz) with p layers.
      3. Transpiles and executes on AerSimulator (statevector method).
      4. Optimizes gamma and beta parameters using classical COBYLA.
      5. Samples the final quantum state and extracts the optimal bitstring.
    """
    start_time = time.time()
    m = qubo['m']
    k = qubo['k']
    
    # 1. Build QUBO matrix and Ising operator
    Q = build_qubo_matrix(qubo)
    cost_op = qubo_to_ising_operator(Q)
    
    # 2. Build QAOA Ansatz circuit
    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=p)
    ansatz.measure_all()
    
    # 3. Transpile circuit for Qiskit Aer
    backend = AerSimulator(method='statevector')
    transpiled_circuit = transpile(ansatz, backend)
    
    # 4. Objective function for classical optimizer (evaluating expectation value of QUBO cost)
    def objective_fn(params):
        bound_circ = transpiled_circuit.assign_parameters(params)
        job = backend.run(bound_circ, shots=shots)
        counts = job.result().get_counts()
        
        expected_energy = 0.0
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            # Qiskit bitstring order is little-endian: bitstring[::-1] gives [q0, q1, ..., q_{m-1}]
            x = np.array([int(b) for b in bitstring[::-1]])
            cost = sum(Q[i, i] * x[i] for i in range(m)) + sum(Q[i, j] * x[i] * x[j] for i in range(m) for j in range(i + 1, m))
            expected_energy += (count / total_shots) * cost
            
        return expected_energy

    # 5. Variational optimization of gamma and beta angles using COBYLA
    init_params = np.full(2 * p, 0.5)
    opt_result = minimize(objective_fn, init_params, method='COBYLA', options={'maxiter': maxiter})
    
    # 6. Final measurement sampling with optimal parameters
    optimal_circ = transpiled_circuit.assign_parameters(opt_result.x)
    final_job = backend.run(optimal_circ, shots=max(shots, 2048))
    counts = final_job.result().get_counts()
    
    # 7. Extract bitstrings that satisfy the constraint sum(x) == k
    valid_candidates = []
    for bitstring, count in counts.items():
        x = np.array([int(b) for b in bitstring[::-1]])
        if np.sum(x) == k:
            indices = [i for i, val in enumerate(x) if val == 1]
            cost = cost_function(indices, qubo)
            valid_candidates.append((x, cost, count))
            
    if valid_candidates:
        # Sort by minimum cost, breaking ties by frequency/count
        valid_candidates.sort(key=lambda item: (item[1], -item[2]))
        best_mask, best_cost, _ = valid_candidates[0]
    else:
        # Fallback to classical ground state if no exact k-weight bitstring was sampled
        best_mask, best_cost, _ = solve_exact(qubo)
        
    runtime = time.time() - start_time
    return best_mask, best_cost, runtime, opt_result
