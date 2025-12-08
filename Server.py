import time
import io
import base64
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
from flask import Flask, request, jsonify

# =============================
# Flask App
# =============================
app = Flask(__name__)

# =============================
# Shared State
# =============================
latest_rr = None
latest_plot = None
latest_ecg_numbers = []

ecg_buffer = []   # incoming ECG packets from IDE

# =============================
# ECG Processing Settings
# =============================
fs = 50
window_sec = 30
window_samples = fs * window_sec      # 1500
stride_samples = 150                  # 3 seconds

# =============================
# NeuroKit Worker Thread
# =============================
def neurokit_worker():
    global latest_rr, latest_plot, latest_ecg_numbers, ecg_buffer

    print("[NeuroKit] Worker started.")

    raw_buffer = []
    last_idx = 0
    new_samples_counter = 0

    while True:
        time.sleep(0.1)

        # -----------------------------
        # Pull ONLY new ECG samples
        # -----------------------------
        if len(ecg_buffer) > last_idx:
            new_chunk = ecg_buffer[last_idx:]
            last_idx = len(ecg_buffer)

            new_vals = [d["ecg"] for d in new_chunk]
            raw_buffer.extend(new_vals)
            new_samples_counter += len(new_vals)

        # Trim buffer
        if len(raw_buffer) > window_samples * 2:
            raw_buffer = raw_buffer[-window_samples * 2:]

        # -----------------------------
        # Process Every 3 Seconds (150 samples)
        # -----------------------------
        if len(raw_buffer) >= window_samples and new_samples_counter >= stride_samples:

            new_samples_counter -= stride_samples

            window = np.array(raw_buffer[-window_samples:], dtype=float)
            window = pd.Series(window).interpolate().bfill().to_numpy()

            # ---- Respiration Extraction ----
            try:
                edr = nk.ecg_rsp(window, sampling_rate=fs)
                rr_series = nk.rsp_rate(edr, sampling_rate=fs)
                rr_vals = [x for x in rr_series if x > 0]

                latest_rr = float(np.mean(rr_vals)) if rr_vals else 0.0

            except Exception as e:
                print("NeuroKit Error:", e)
                latest_rr = None

            # ---- Last ECG Samples for Streaming ----
            latest_ecg_numbers[:] = window[-500:].tolist()

            # ---- Generate Plot ----
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

# =============================
# API ENDPOINTS
# =============================

@app.route("/upload_ecg", methods=["POST"])
def upload_ecg():
    """
    Receives 50 ECG samples every 3 seconds
    """
    global ecg_buffer

    data = request.json
    samples = data.get("samples", [])

    for v in samples:
        ecg_buffer.append({"ecg": float(v)})

    return jsonify({"status": "ok", "received": len(samples)})


@app.route("/rr", methods=["GET"])
def get_rr():
    return jsonify({"rr": latest_rr})


@app.route("/ecg_plot", methods=["GET"])
def get_plot():
    return jsonify({"image": latest_plot})


@app.route("/ecg_stream", methods=["GET"])
def get_ecg_numbers():
    return jsonify({"ecg": latest_ecg_numbers})


# =============================
# Start Worker Thread
# =============================
threading.Thread(target=neurokit_worker, daemon=True).start()

# =============================
# Run Server
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
