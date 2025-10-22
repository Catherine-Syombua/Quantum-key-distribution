# Quantum-key-distribution
A streamlit-based simulation of Dense Quantum Key distribution using superdense coding with Qiskit
[qkd_dense.py](https://github.com/user-attachments/files/23061789/qkd_dense.py)
import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.quantum_info import state_fidelity, DensityMatrix

# Simulator instance (can be overridden)
SIM = AerSimulator()

def build_noise_model(depolarizing_prob=0.0, bit_flip_prob=0.0):
    """Return a qiskit NoiseModel with simple depolarizing and bit-flip channels."""
    nm = NoiseModel()
    if depolarizing_prob > 0:
        # single qubit depolarizing error
        de = depolarizing_error(depolarizing_prob, 1)
        nm.add_all_qubit_quantum_error(de, ['u1', 'u2', 'u3', 'x', 'h', 'id', 'cx'])
    if bit_flip_prob > 0:
        bf = pauli_error([('X', bit_flip_prob), ('I', 1 - bit_flip_prob)])
        nm.add_all_qubit_quantum_error(bf, ['x', 'u1', 'u2', 'u3'])
    return nm

def create_bell_pair_circuit():
    """Return a circuit that creates |Φ+> on qubits 0 (Alice) and 1 (Bob)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def apply_dense_encoding(qc, alice_qubit_index, bits):
    """Apply encoding to Alice's qubit to encode 2 classical bits (string '00','01','10','11')."""
    if bits == '00':
        pass  # I
    elif bits == '01':
        qc.x(alice_qubit_index)
    elif bits == '10':
        qc.z(alice_qubit_index)
    elif bits == '11':
        qc.x(alice_qubit_index)
        qc.z(alice_qubit_index)
    else:
        raise ValueError("bits must be one of '00','01','10','11'")

def bell_measurement_circuit():
    """Return operations (as a QuantumCircuit) that map Bell basis to computational basis for measurement."""
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc

def simulate_superdense_round(bits_to_send, include_eve=False, eve_strategy='measure_resend',
                             noise_params=None, simulator=SIM):
    """
    Simulate a single superdense encoding round.
    - bits_to_send: '00','01','10' or '11'
    - include_eve: bool, whether Eve intercepts
    - eve_strategy: 'measure_resend' (random basis)
    - noise_params: dict { 'depolarizing': p, 'bit_flip': p }
    Returns: dict with keys:
      'alice_bits', 'bob_bits', 'detected_eve' (bool flag if check bits mismatched), 'raw_measure',
      'final_bits' (2-bit string Bob decoded), 'ideal_bits' (what should be received)
    """
    # Build initial Bell creation + encoding + optional Eve + Bell measurement
    # We'll assemble a complete circuit for the system qubits; for Eve intercept we'll emulate measurement & resend steps
    
    noise_model = None
    if noise_params:
        noise_model = build_noise_model(noise_params.get('depolarizing', 0.0),
                                        noise_params.get('bit_flip', 0.0))
    # Create the circuit that prepares entanglement and encodes
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    # Encode on Alice's qubit (qubit 0)
    apply_dense_encoding(qc, 0, bits_to_send)

    # If Eve intercepts, we simulate her by measuring Alice's qubit in a random basis and re-preparing a qubit
    # Approach: append operations to qc that simulate Eve measuring qubit 0 and re-preparing accordingly.
    # We'll introduce an ancilla qubit for re-preparation to avoid overwriting Bob's qubit (but for simplicity here we re-use qubit 0)
    if include_eve:
        # Choose Eve basis randomly per round: 'Z' or 'X'
        eve_basis = random.choice(['Z', 'X'])
        if eve_basis == 'X':
            qc.h(0)
        qc.measure(0, 0)  # measure into classical bit 0
        # After measurement, re-prepare qubit state according to measurement outcome:
        # We'll use conditional operations in classical bits to re-prepare; but aer's classical conditioning
        # in circuits is limited; instead, simpler approach: we just run measurement as an intermediate job to get outcome,
        # then build a new circuit continuing from post-measure state. For simplicity of deterministic single-shot simulation,
        # we will split this into two simulator calls: measure (to get eve_result), then rebuild the remaining circuit.
        # So we stop here and run the first part.
        qc_pre = qc.copy()
        qc_pre.measure_all()  # ensure measurements are appended for simulator run
        job = simulator.run(transpile(qc_pre, simulator), shots=1, memory=True,
                            noise_model=noise_model).result()
        mem = job.get_memory()[0]
        # memory returns bitstring of length 2 for measured classical registers; since we used 2 classical bits we expect 'b1b0'
        # But to be safe parse last bit (associated with qubit 0 measurement)
        # In our qc_pre measurement we measured both (because measure_all), so we pick the right bit.
        # The memory string order for measure_all might be reversed; better to read as int and check last char.
        if len(mem) >= 1:
            eve_meas = mem[-1]  # '0' or '1'
        else:
            eve_meas = '0'
        # Rebuild a new circuit continuing: create fresh Bell pair? No — Eve has disturbed Alice's qubit.
        # We'll create a new circuit where Bob's qubit is initialized according to what he still has, and Alice's qubit is prepared from Eve's resend.
        # Simpler: we reconstruct the post-measure state: Eve will resend |0> or |1> (if measured in Z) or |+>/|-> if measured in X.
        # For practicality, we will prepare a circuit that sets qubit 0 according to Eve's measurement outcome.
        qc2 = QuantumCircuit(2, 2)
        # Prepare Bob's qubit: we cannot reconstruct Bob's exact state easily without full statevector tracking; instead,
        # we will treat Eve's action as: she replaces Alice's qubit with a fresh qubit prepared in her measurement outcome,
        # and Bob retains his qubit from the entanglement (but that entanglement is now broken).
        # For simulation, we'll initialize qubit 1 to |0>, then apply nothing (we accept that entanglement lost).
        # Prepare Alice's qubit according to eve_basis and eve_meas:
        if eve_basis == 'Z':
            if eve_meas == '1':
                qc2.x(0)
        else:  # X basis measured
            if eve_meas == '1':
                qc2.x(0)
            qc2.h(0)  # prepare |+> if measured 0 -> h|0>, if measured 1 -> x then h -> |->

        # Bob's qubit we will init to |0> and then (optionally) apply nothing - it's a simple lowered-fidelity model.
        # In practice this models an intercept-resend attack which destroys correlations.
        # Continue with Bell measurement
        qc2.cx(0, 1)
        qc2.h(0)
        qc2.measure([0, 1], [0, 1])
        qct = qc2
        job2 = simulator.run(transpile(qct, simulator), shots=1, memory=True,
                             noise_model=noise_model).result()
        mem2 = job2.get_memory()[0]
        # mem2 is bitstring of classical registers; order might be 'b1b0' so reverse to read as (bit0,bit1)
        decoded = mem2[::-1][:2]  # take two bits, reversed
        bob_bits = decoded
        # For simplicity, we set ideal bits to original bits_to_send (what Alice attempted)
        return {
            'alice_bits': bits_to_send,
            'bob_bits': bob_bits,
            'eve_basis': eve_basis,
            'eve_meas': eve_meas,
            'noise_model': noise_model,
            'final_bits': bob_bits,
            'ideal_bits': bits_to_send
        }

    # No Eve: construct measurement and run
    # Append Bell measurement
    qc.cx(0,1)
    qc.h(0)
    qc.measure([0,1], [0,1])
    job = simulator.run(transpile(qc, simulator), shots=1, memory=True,
                        noise_model=noise_model).result()
    mem = job.get_memory()[0]
    # Reverse bit order to read as [bit0, bit1]
    decoded = mem[::-1][:2]
    return {
        'alice_bits': bits_to_send,
        'bob_bits': decoded,
        'final_bits': decoded,
        'ideal_bits': bits_to_send,
        'noise_model': noise_model
    }

def run_dense_qkd_rounds(num_rounds=10, eve_prob=0.0, noise_params=None, simulator=SIM, check_fraction=0.5):
    """
    Run many rounds and perform sifting + QBER estimation.
    - eve_prob: probability that a given round includes Eve intercept (simulates partial Eve presence)
    - noise_params: dict passed to simulate_superdense_round
    - check_fraction: fraction of matched rounds used for QBER estimation
    Returns:
      dict with lists and aggregated metrics
    """
    alice_list = []
    bob_list = []
    eve_info = []
    rounds_data = []

    for i in range(num_rounds):
        bits = random.choice(['00','01','10','11'])
        include_eve = random.random() < eve_prob
        res = simulate_superdense_round(bits, include_eve=include_eve, noise_params=noise_params, simulator=simulator)
        rounds_data.append(res)
        alice_list.append(bits)
        bob_list.append(res['final_bits'])
        eve_info.append(include_eve)

    # Sifting: in dense scheme we do not have basis mismatch as in BB84, but we do security checks using randomized check rounds.
    # Choose indices to check
    indices = list(range(num_rounds))
    random.shuffle(indices)
    num_check = max(1, int(check_fraction * num_rounds))
    check_indices = indices[:num_check]
    data_indices = indices[num_check:]

    # Compute QBER on check bits: compare Alice bits to Bob bits
    errors = 0
    for idx in check_indices:
        if alice_list[idx] != bob_list[idx]:
            errors += 1
    qber = errors / len(check_indices) if check_indices else 0.0

    # Final raw key: concatenation of bits from non-checked rounds that matched
    raw_key_bits = []
    for idx in data_indices:
        if alice_list[idx] == bob_list[idx]:
            raw_key_bits.append(bob_list[idx])
    # key rate = (2 * number_of_successful_rounds) / number_qubits_sent (here each round sends 1 qubit, encodes 2 bits)
    successful = sum(1 for idx in data_indices if alice_list[idx] == bob_list[idx])
    key_bits_length = 2 * successful  # each successful round yields 2 classical bits
    key_rate = key_bits_length / num_rounds

    return {
        'rounds_data': rounds_data,
        'alice_list': alice_list,
        'bob_list': bob_list,
        'eve_present': eve_info,
        'check_indices': check_indices,
        'data_indices': data_indices,
        'qber': qber,
        'raw_key_blocks': raw_key_bits,
        'key_bits_length': key_bits_length,
        'key_rate': key_rate
    }


