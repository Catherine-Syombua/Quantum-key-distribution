import streamlit as st
import pandas as pd
import numpy as np
from qkd_dense import run_dense_qkd_rounds, simulate_superdense_round
from viz import plot_qber_history, plot_keyrate_vs_noise, make_rounds_table, csv_from_df

st.set_page_config(page_title="Dense QKD (Superdense) Demo", layout="wide")

st.title("Dense QKD — Superdense Coding based Key Distribution")
st.markdown("""
This demo implements a Dense QKD variant based on **superdense coding**:
- Alice and Bob share entangled pairs.
- Alice encodes **2 classical bits** by applying I/X/Z/XZ to her qubit.
- Alice sends her qubit to Bob, who performs a Bell measurement to decode the 2 bits.
- We include optional Eve (intercept-resend) and channel noise models.
""")

# Sidebar controls
st.sidebar.header("Simulation Controls")
num_rounds = st.sidebar.slider("Number of rounds", 4, 200, value=20, step=1)
eve_prob = st.sidebar.slider("Probability Eve intercepts each round", 0.0, 1.0, value=0.2, step=0.05)
depolarizing = st.sidebar.slider("Depolarizing probability (channel noise)", 0.0, 0.5, value=0.0, step=0.01)
bit_flip = st.sidebar.slider("Bit-flip probability", 0.0, 0.5, value=0.0, step=0.01)
check_fraction = st.sidebar.slider("Fraction of rounds used for QBER checks", 0.1, 0.9, value=0.5, step=0.05)
show_rounds = st.sidebar.checkbox("Show per-round table", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Advanced:**")
show_circuit_samples = st.sidebar.checkbox("Show sample circuit diagrams (matplotlib)", value=False)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        noise_params = {'depolarizing': depolarizing, 'bit_flip': bit_flip}
        results = run_dense_qkd_rounds(num_rounds=num_rounds, eve_prob=eve_prob, noise_params=noise_params, check_fraction=check_fraction)
        st.success("Simulation complete")

        # Metrics
        st.subheader("Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("QBER (check rounds)", f"{results['qber']*100:.2f}%")
        col2.metric("Raw key blocks (2-bit blocks)", len(results['raw_key_blocks']))
        col3.metric("Key length (bits)", results['key_bits_length'])
        st.markdown(f"**Key rate**: {results['key_rate']:.3f} bits per qubit (expected ~2 for perfect runs)")

        # Round details
        if show_rounds:
            st.subheader("Per-round results (first 50 rounds shown)")
            df_rounds = make_rounds_table(results['rounds_data'])
            st.dataframe(df_rounds.head(50))

            csv = csv_from_df(df_rounds)
            st.download_button("Download rounds CSV", csv, file_name="dense_qkd_rounds.csv", mime="text/csv")

        # Visualizations
        st.subheader("Visualizations")
        # For demo: simulate multiple experiments at different noise levels to plot Key Rate vs Noise if depolarizing > 0
        if depolarizing > 0:
            st.markdown("Key rate vs depolarizing noise (quick sweep)")
            noise_levels = list(np.linspace(0, depolarizing, num=6))
            key_rates = []
            for p in noise_levels:
                r = run_dense_qkd_rounds(num_rounds=40, eve_prob=eve_prob, noise_params={'depolarizing': p, 'bit_flip': bit_flip}, check_fraction=check_fraction)
                key_rates.append(r['key_rate'])
            fig = plot_keyrate_vs_noise(noise_levels, key_rates)
            st.pyplot(fig)

        st.markdown("QBER over recent run (single value shown across experiments)")
        qber_list = [results['qber']]
        fig2 = plot_qber_history(qber_list)
        st.pyplot(fig2)

        # Show sample circuits if requested
        if show_circuit_samples:
            st.subheader("Sample quantum circuit diagrams (text)")
            st.code("""
# Bell pair + encoding schematic (Alice applies gates on qubit 0)
qc.h(0)
qc.cx(0,1)
# Alice encodes 2-bit value using I, X, Z, XZ on qubit 0
# Then Alice sends qubit 0 to Bob (simulated)
# Bob does: qc.cx(0,1); qc.h(0); qc.measure([0,1],[0,1])
""")

        # Display raw key blocks
        st.subheader("Raw Key Blocks (2-bit blocks from successful non-check rounds)")
        st.write(results['raw_key_blocks'])

        # Simple interpretation
        st.markdown("### Interpretation")
        if results['qber'] > 0.2:
            st.error("High QBER detected — likely an eavesdropper or severe noise. Discard key.")
        elif results['qber'] > 0.05:
            st.warning("Moderate QBER — may need error correction / privacy amplification.")
        else:
            st.success("Low QBER — key considered usable after privacy amplification.")

        # Offer simple OTP encryption demo
        st.subheader("Quick Demo: Use key to encrypt a short message (XOR OTP on bits)")
        msg = st.text_input("Message to encrypt (short ASCII):", value="HELLO")
        if st.button("Encrypt message using raw key"):
            # Build raw key bitstring from raw_key_blocks
            if len(results['raw_key_blocks']) == 0:
                st.error("No raw key available — run more rounds or reduce check fraction.")
            else:
                # Flatten 2-bit blocks into bitstring
                bitstring = ''.join(results['raw_key_blocks'])
                # Convert to bytes stream by grouping 8 bits
                # If not enough bits, pad with zeros
                needed_bits = len(msg) * 8
                bits = bitstring.replace(' ', '')
                if len(bits) < needed_bits:
                    bits = bits.ljust(needed_bits, '0')
                # XOR each character
                cipher_bytes = []
                for i, ch in enumerate(msg.encode('utf-8')):
                    # take 8 bits slice
                    b = bits[i*8:(i+1)*8]
                    key_byte = int(b, 2)
                    cipher_bytes.append(ch ^ key_byte)
                cipher = bytes(cipher_bytes)
                st.code(f"Cipher (hex): {cipher.hex()}")
                # Also show decryption
                decrypted = bytes([c ^ int(bits[i*8:(i+1)*8], 2) for i, c in enumerate(cipher)])
                st.code(f"Decrypted text: {decrypted.decode('utf-8', errors='ignore')}")



