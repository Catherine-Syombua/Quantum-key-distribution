# Team Members
Sharon Wambua
Catherine Syombua
Joan Kirui

# Concept Description
This project solves the problem of secure key distribution - how two parties (Alice and Bob) can create a shared secret key over an insecure channel while detecting whether a third party (Eve) has been eavesdropping.

It implements and visualizes a Superdense Quantum Key Distribution (Dense QKD) protocol; an  extension of quantum key distribution that combines superdense coding, entanglement, and quantum noise simulation to demonstrate secure key sharing.

This system uses entangled qubit pairs (Bell states) and superdense encoding to transmit two classical bits per qubit, effectively doubling the key rate while maintaining eavesdropper detection through Quantum Bit Error Rate (QBER) estimation.

The project is built as an interactive Streamlit web app, enabling real-time simulation, visualization, and analysis of quantum communication security under varying conditions including noise and interception.

# How the solution works
This project simulates how Alice and Bob can securely share a secret key using superdense coding, an advanced quantum communication technique.

In superdense coding, two classical bits of information are transmitted using only one qubit, made possible through quantum entanglement.

1. Entanglement Creation:
The system generates entangled qubit pairs (Bell states). Alice keeps one qubit, and Bob holds the other. These qubits are quantumly linked — any change to one affects the other.

2. Encoding:
Alice encodes her two classical bits (00, 01, 10, or 11) by applying a specific quantum operation (I, Z, X, or XZ) to her qubit. Each operation changes the shared entangled state uniquely.

3. Transmission & Decoding:
Alice sends her qubit to Bob through a simulated quantum channel. Bob then performs a Bell-state measurement on both qubits to decode Alice’s original two-bit message.

4. Security Verification (QBER):
To check if the channel is secure, Alice and Bob publicly compare a subset of rounds. The Quantum Bit Error Rate (QBER) measures how many bits differ.
* Low QBER → secure channel
* High QBER → possible noise or eavesdropping (Eve)

5. Key Generation:
After removing the checked rounds, the remaining matching results form the shared secret key. This key can then be used for classical encryption or secure communication.

The app lets users simulate noise and eavesdropping, visualize results, and download reports, showing how entanglement enables secure and efficient quantum key exchange.


# Core Features
Entanglement-based superdense encoding
→ Two classical bits encoded in one qubit via Bell pairs

Quantum noise simulation
→ Add bit-flip, phase-flip, or depolarizing noise

Eavesdropper (Eve) simulation
→ Random measurement disturbance and re-preparation

Sifting and QBER estimation
→ Detect potential eavesdropping by comparing test bits

Key metrics dashboard
→ Key rate, error rate (QBER), and secure key length

Circuit visualization
→ View encoding and decoding quantum circuits

Downloadable reports
→ Export run results as CSV for analysis or documentation

# Demo
1. Simulation Setup
Choose the number of rounds, noise level, and whether Eve is present on the app. Each round represents one instance of Alice sending two bits using superdense coding.
* Alice encodes random 2-bit messages.
* Bob decodes them using Bell-state measurements.
* Some rounds are randomly marked as “check” rounds for security testing.
	Check rounds are used to estimate QBER.
	Non-check rounds form the shared raw key.

2. QBER Estimation
The Quantum Bit Error Rate (QBER) measures how often Alice’s and Bob’s bits disagree in the check rounds

Example: 0 mismatches out of 4 checks → QBER = 0%
Interpretation:
* QBER < 5% → Channel is secure
* 5–20% → Possibly noisy
* >0% → Potential eavesdropping detected

3. Key Extraction & Rate
Only non-check, correctly decoded rounds are kept as the raw key.
If two such rounds remain with perfect decoding (e.g., 01 and 11), the shared raw key becomes 0111.
Key rate = (usable bits / total transmitted qubits).
Example: 4 bits / 20 qubits = 0.20 bits per qubit.

4. Outcome Analysis
* Low QBER: Secure channel, no Eve detected.
* Moderate QBER: Channel noise or small interference.
* High QBER: Strong evidence of interception — abort communication.

5. Report and Visualization
The app provides:
* A QBER vs. noise plot for performance insight.
* Key rate and error trends over multiple runs.
* Downloadable CSV report with all simulation data per round.


