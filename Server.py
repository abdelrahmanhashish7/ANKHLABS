from flask import Flask, request, jsonify, send_file
import io
import base64
import threading
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk

app = Flask(__name__)

# -------------------------------------------------------
# SHARED BUFFERS (server + NeuroKit share these)
# -------------------------------------------------------
ecg_buffer = []              # [{"ecg": x, "timestamp": t}, ...]
glucose_buffer = []
latest_glucose = {"value": None, "timestamp": None}

# NeuroKit processed outputs
latest_rr = None
latest_plot = None
latest_ecg_numbers = []   # last 500 processed ECG samples


# -------------------------------------------------------
# FLASK ROUTES
# -------------------------------------------------------

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    if not data or "ecg" not in data:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    ecg = data.get("ecg")
    glucose = data.get("glucose")
    timestamp = data.get("timestamp")

    # Store ECG sample
    ecg_buffer.append({"ecg": ecg, "timestamp": timestamp})

    # Store glucose sample (if exists)
    if glucose is not None:
        latest_glucose["value"] = glucose
        latest_glucose["timestamp"] = timestamp
        glucose_buffer.append({"glucose": glucose, "timestamp": timestamp})

    return jsonify({"status": "ok"}), 200


@app.route('/ecg', methods=['GET'])
def get_ecg():
    return jsonify(ecg_buffer[-500:])   # last 500 raw samples


@app.route('/clear', methods=['POST'])
def clear_buffer():
    ecg_buffer.clear()
    glucose_buffer.clear()
    return jsonify({"status": "cleared"})


@app.route('/resp_rate', methods=['GET'])
def show_resp_rate():
    return jsonify({"resp_rate": latest_rr})


@app.route('/ecg_plot', methods=['GET'])
def show_ecg_plot():
    if latest_plot is None:
        return "No plot available", 404

    img_bytes = base64.b64decode(latest_plot)
    return send_file(io.BytesIO(img_bytes), mimetype="image/png")


@app.route('/ecgnumbers', methods=['GET'])
def show_ecg_numbers():
    return jsonify({"numbers": latest_ecg_numbers})


@app.route('/glucose', methods=['GET'])
def show_glucose():
    if latest_glucose["value"] is None:
        return jsonify({"glucose": None}), 404

    return jsonify(latest_glucose)


@app.route('/glucose_history', methods=['GET'])
def glucose_history():
    return jsonify({"glucose_history": glucose_buffer})


@app.route('/test')
def test():
    return "Server + NeuroKit running"


# -------------------------------------------------------
# NEUROKIT REAL-TIME PROCESSING THREAD (INSIDE SERVER)
# -------------------------------------------------------

def neurokit_worker():
    global latest_rr, latest_plot, latest_ecg_numbers

    fs = 50                 # sampling frequency (change to your ESP32 sampling rate)
    window_sec = 30
    window_samples = fs * window_sec

    raw_buffer = []

    print("[NeuroKit] Worker started.")

    while True:
        time.sleep(1)  # run every 1 sec

        # Add last ECG values from buffer
        if len(ecg_buffer) >= 1:
            new_samples = [d["ecg"] for d in ecg_buffer[-200:]]  # take last ~200 samples
            raw_buffer.extend(new_samples)

        # Keep last ~60 seconds max
        if len(raw_buffer) > window_samples * 2:
            raw_buffer = raw_buffer[-window_samples * 2:]

        # Need a full window for processing
        if len(raw_buffer) < window_samples:
            continue

        # Get fixed window of 30s
        window = np.array(raw_buffer[-window_samples:], dtype=float)

        # Clean missing values
        window = pd.Series(window).interpolate().bfill().to_numpy()

        # --------------------------
        # RESPIRATION EXTRACTION
        # --------------------------
        try:
            edr = nk.ecg_rsp(window, sampling_rate=fs)
            rr_series = nk.rsp_rate(edr, sampling_rate=fs)

            # Average only positive values
            rr_vals = [v for v in rr_series if v > 0]
            latest_rr = float(np.mean(rr_vals)) if rr_vals else 0.0

        except Exception as e:
            print("[NeuroKit] Respiration error:", e)
            latest_rr = None

        # --------------------------
        # ECG LAST NUMBERS
        # --------------------------
        latest_ecg_numbers = window[-500:].tolist()

        # --------------------------
        # PLOT GENERATION
        # --------------------------
        try:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(window[-500:])
            ax.set_title("ECG (last 10 seconds)")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            latest_plot = base64.b64encode(buf.read()).decode()

            plt.close(fig)

        except Exception as e:
            print("[NeuroKit] Plot error:", e)
            latest_plot = None


# Start the background NeuroKit thread when server starts
threading.Thread(target=neurokit_worker, daemon=True).start()


# -------------------------------------------------------
# START FLASK SERVER
# -------------------------------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
