from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
import numpy as np
import matplotlib.pyplot as plt

def swap_exp(delta_t):
    # a tiny circuit that applies the eponential of the SWAP gate
    # since SWAP can be decomposed as S = deltat(II+XX+YY+ZZ)/2 (tensor products between each II XX YY ZZ ofc)

    qc = QuantumCircuit(2) #two bit circuit

    # Apply RXX, RYY, and RZZ with delta_t (note: factor 0.5 absorbed in angle)
    qc.append(RXXGate(2 * delta_t), [0, 1])
    qc.append(RYYGate(2 * delta_t), [0, 1])
    qc.append(RZZGate(2 * delta_t), [0, 1])

    return qc

def densitymatrixexp(t=1.0, steps=10, rho_copies=3, sigma=None, rho=None):
    """
    Constructs a circuit to simulate the evolution e^{-i ρ t} σ e^{i ρ t}
    using density matrix exponentiation with repeated SWAP operations.

    Args:
        t (float): Total evolution time.
        steps (int): Number of Trotter steps.
        rho_copies (int): Number of copies of the density matrix ρ.
        sigma (QuantumCircuit or Statevector or None): Initial state for σ (target system).
        rho (QuantumCircuit or Statevector or None): State representing ρ (used in copies).
                                                     If None, defaults to |+⟩.
    Returns:
        QuantumCircuit: The constructed circuit simulating e^{-iρt}σe^{iρt}.
    """

    
    # Small time step per Trotterized evolution
    delta_t = t / steps

    # Total number of qubits:
    # - Qubit 0: holds the σ state (the "system" we evolve)
    # - Qubits 1 to n: hold copies of ρ (the "environment")
    total_qubits = 1 + rho_copies
    qr = QuantumRegister(total_qubits)
    qc = QuantumCircuit(qr)


    # STEP 1: Initialize σ on qubit 0
    if sigma is None:
        pass  # Default is |0⟩ (no action needed)
    elif isinstance(sigma, Statevector):
        qc.initialize(sigma.data, qr[0])  # set qubit 0 to the statevector
    elif isinstance(sigma, DensityMatrix):
        qc.initialize(sigma.to_statevector().data, qr[0])
    elif isinstance(sigma, QuantumCircuit):
        qc.compose(sigma, qubits=[qr[0]], inplace=True)
    else:
        raise ValueError("sigma must be a Statevector, DensityMatrix, or QuantumCircuit")

    # STEP 2: Initialize all copies of ρ on qubits 1 to rho_copies
    for i in range(1, total_qubits):
        if rho is None:
            qc.h(qr[i])  # Default to |+⟩
        elif isinstance(rho, Statevector):
            qc.initialize(rho.data, qr[i])  # directly set the state
        
        #the desitymatrix part isnt really working, avoid using densitymatrix objects for now
        elif isinstance(rho, DensityMatrix):
            qc.initialize(rho.to_statevector().data, qr[i])    
        #elif isinstance(rho, DensityMatrix):
        # Try to convert to a pure state (only works if rank 1)
        #    try:
        #        sv = rho.to_statevector()
        #        qc.initialize(sv.data, qr[i])
        #    except Exception:
        #        raise ValueError("Density matrix is not pure. Cannot initialize qubit directly.")
        elif isinstance(rho, QuantumCircuit):
            qc.compose(rho, qubits=[qr[i]], inplace=True)
        else:
            raise ValueError("rho must be a Statevector, DensityMatrix, or QuantumCircuit")


    # STEP 3: Apply the SWAP exponentials
    # Each block approximates exp(-i S δt), where S is the SWAP operator
    for _ in range(steps):
        for i in range(1, total_qubits):
            # Create a two-qubit gate implementing exp(-i SWAP δt)
            block = swap_exp(delta_t)
            block.qregs = []  # Clear any existing registers to allow appending
            instr = block.to_instruction(label=f"exp(-i S_{i} dt)")

            # Append the SWAP exponential between σ (qubit 0) and the i-th ρ copy
            qc.append(instr, [0, i])

    return qc

def trace_out_rhos(qc):
    """
    Simulates the circuit and traces out all qubits except the first (σ).
    
    Args:
        qc (QuantumCircuit): Circuit containing σ at position 0 and any number of copies of ρ on others.
    
    Returns:
        DensityMatrix: The reduced density matrix of σ after tracing out the ρ copies.
    """
    # Simulate the full statevector
    full_sv = Statevector.from_instruction(qc)
    
    # Total number of qubits
    num_qubits = qc.num_qubits

    # Trace out all qubits except qubit 0 (σ)
    reduced_dm = partial_trace(full_sv, list(range(1, num_qubits)))

    return reduced_dm


# Example rho =|-⟩
rho1 = Statevector.from_label('-')
rho_matrix = [[1/2,-1/2], [-1/2, 1/2]]

# Print eigenvalues
rho_evals, _ = np.linalg.eigh(rho_matrix)
print("ρ eigenvalues:", rho_evals)

# σ = |+⟩ = (|0⟩ + |1⟩)/√2
sigma1 = DensityMatrix(Statevector.from_label('+'))

qc = densitymatrixexp(t=1.0, steps=30, rho_copies=5, sigma=sigma1, rho=rho1)
reduced_sigma = trace_out_rhos(qc)

# Get eigenvalues of final reduced sigma
sigma_evals, _ = np.linalg.eigh(reduced_sigma.data)
print("σ evolved eigenvalues:", sigma_evals)
