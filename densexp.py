from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, StatePreparation
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
        sigma (Statevector): Initial state for σ (target system).
        rho (Statevector): State representing ρ (used in copies).
    Returns:
        QuantumCircuit: The constructed circuit simulating e^{-iρt}σe^{iρt}.
    """

    
    # Small time step per Trotterized evolution
    delta_t = t / steps

    # Determine the number of qubits for sigma and rho
    sigma_qubits = int(np.log2(len(sigma))) if isinstance(sigma, Statevector) else 1
    rho_qubits = int(np.log2(len(rho))) if isinstance(rho, Statevector) else 1

    total_qubits = sigma_qubits + rho_copies * rho_qubits
    qr = QuantumRegister(total_qubits)
    qc = QuantumCircuit(qr)


    # STEP 1: Initialize sigma
    if isinstance(sigma, Statevector):
        qc.initialize(sigma.data, qr[0:sigma_qubits])
    #elif isinstance(sigma, QuantumCircuit):
    #    sv = Statevector.from_instruction(sigma)
    #    sigma_qubits = int(np.log2(len(sv)))
    #    qc.initialize(sv.data, qr[0:sigma_qubits])
    elif sigma is None:
        pass
    else:
        raise ValueError("sigma must be Statevector")

    # STEP 2: Initialize rho copies
    for i in range(rho_copies):
        q_start = sigma_qubits + i * rho_qubits
        q_end = q_start + rho_qubits
        if isinstance(rho, Statevector):
            qc.initialize(rho.data, qr[q_start:q_end])
        #elif isinstance(rho, QuantumCircuit):
        #    sv = Statevector.from_instruction(rho)
        #    qc.compose(rho, qubits=qr[q_start:q_end], inplace=True)
        elif rho is None:
            pass
        else:
            raise ValueError("rho must be Statevector")

    # STEP 3: Apply SWAP exponentials between sigma and each rho copy
    for _ in range(steps):
        for i in range(rho_copies):
            sigma_idx = list(range(0, sigma_qubits))
            rho_idx = list(range(sigma_qubits + i * rho_qubits,
                                 sigma_qubits + (i + 1) * rho_qubits))

            for s, r in zip(sigma_idx, rho_idx):
                block = swap_exp(delta_t)
                block.qregs = []
                qc.append(block.to_instruction(), [s, r])

    return qc

def ptr1(qc, sigma=None): #not sure we will be needing this bad boy
    """
    Simulates the circuit and traces out all qubits except those corresponding to σ in the first qbits.

    Args:
        qc (QuantumCircuit): Circuit with σ in the first qubits.
        sigma (Statevector or DensityMatrix): The initial σ state (used to infer width).

    Returns:
        DensityMatrix: The reduced density matrix of σ after tracing out the ρ copies.
    """
    if sigma is None:
        raise ValueError("You must provide the initial σ state to determine its qubit width.")

    # Infer number of qubits used for σ from the state
    if isinstance(sigma, Statevector) or isinstance(sigma, DensityMatrix):
        dim = sigma.dim  # this is 2**n
        sigma_qubits = int(np.log2(dim))
    else:
        raise ValueError("sigma must be a Statevector or DensityMatrix")

    # Simulate the full statevector
    full_sv = Statevector.from_instruction(qc)

    # Trace out all but σ qubits (which are assumed to be in the first `sigma_qubits`)
    trace_out_indices = list(range(sigma_qubits, qc.num_qubits))

    reduced_dm = partial_trace(full_sv, trace_out_indices)

    return reduced_dm

sigma1 = Statevector.from_label('00')

rho_matrix = np.array([[1,2,3,4],[5,4,6,3],[1,2,3,9],[1,1,1,1]])
rhoamp = rho_matrix.flatten()
rhoamp = rhoamp/np.linalg.norm(rhoamp)
#print(rhoamp)
rho2=Statevector(StatePreparation(rhoamp))

rho_evals, _ = np.linalg.eigh(rho_matrix)
print("ρ eigenvalues:", rho_evals)

qc = densitymatrixexp(t=1.0, steps=100, rho_copies=3, sigma=sigma1, rho=rho2)
reduced_sigma = ptr1(qc, sigma=sigma1)

# Get eigenvalues of final reduced sigma
sigma_evals, _ = np.linalg.eigh(reduced_sigma.data)
print("σ evolved eigenvalues:", sigma_evals)