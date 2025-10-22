import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

def plot_qber_history(qber_values):
    fig, ax = plt.subplots()
    ax.plot(range(len(qber_values)), qber_values, marker='o')
    ax.set_xlabel("Experiment index")
    ax.set_ylabel("QBER")
    ax.set_title("QBER over experiments")
    ax.grid(True)
    return fig

def plot_keyrate_vs_noise(noise_levels, key_rates):
    fig, ax = plt.subplots()
    ax.plot(noise_levels, key_rates, marker='o')
    ax.set_xlabel("Depolarizing probability")
    ax.set_ylabel("Key rate (bits per qubit)")
    ax.set_title("Key Rate vs Channel Noise")
    ax.grid(True)
    return fig

def make_rounds_table(rounds_data):
    # rounds_data: list of dicts returned per round
    rows = []
    for i, r in enumerate(rounds_data):
        rows.append({
            'round': i,
            'alice_bits': r.get('alice_bits'),
            'bob_bits': ''.join(r.get('bob_bits')) if isinstance(r.get('bob_bits'), (list, tuple)) else r.get('bob_bits'),
            'eve_basis': r.get('eve_basis', ''),
            'eve_meas': r.get('eve_meas', '')
        })
    df = pd.DataFrame(rows)
    return df

def csv_from_df(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
