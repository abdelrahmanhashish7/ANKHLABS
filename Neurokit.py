import time
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
import neurokit2 as nk

# -----------------------------
# Shared state (server reads these)
# -----------------------------
latest_rr = None
latest_plot = None
latest_ecg_numbers = []

# ECG processing settings
fs = 50               # sampling frequency
window_sec = 30       # 30s processing window
window_samples = fs * window_sec


def neurokit_worker(ecg_buffer):
    """Background worker that processes ECG continuously."""
    global latest_rr, latest_plot, latest_ecg_numbers

    print("[NeuroKit] Worker started.")

    raw_buffer = []

    while True:
        time.sleep(1)

        # -----------------------------
        # Add new samples
        # -----------------------------
        if len(ecg_buffer) > 0:
            new_samples = [d["ecg"] for d in ecg_buffer[-200:]]
            raw_buffer.extend(new_samples)

        # Keep buffer manageable
        if len(raw_buffer) > window_samples * 2:
            raw_buffer = raw_buffer[-window_samples * 2:]

        # -----------------------------
        # Process once we have enough data
        # -----------------------------
        if len(raw_buffer) >= window_samples:
            window = np.array(raw_buffer[-window_samples:], dtype=float)

            # interpolate missing values
            window = pd.Series(window).interpolate().bfill().to_numpy()

            # ---- Respiration extraction ----
            try:
                edr = nk.ecg_rsp(window, sampling_rate=fs)
                rr_series = nk.rsp_rate(edr, sampling_rate=fs)

                rr_vals = [x for x in rr_series if x > 0]
                latest_rr = float(np.mean(rr_vals)) if rr_vals else 0.0

            except Exception as e:
                print("NeuroKit error:", e)
                latest_rr = None

            # ---- Last ECG values ----
            latest_ecg_numbers = window[-500:].tolist()

            # ---- Generate plot ----
            try:
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.plot(window[-500:])
                ax.set_title("ECG (10 sec)")
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                latest_plot = base64.b64encode(buf.read()).decode()

                plt.close(fig)
            except:
                latest_plot = None
