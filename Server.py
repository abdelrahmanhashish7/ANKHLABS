from flask import Flask, request, jsonify
import threading
import time
import numpy as np
import pandas as pd
import neurokit2 as nk

app = Flask(__name__)

# ======================================================
# GLOBAL SHARED STATE
# ======================================================
ecg_buffer = []                  # raw ECG samples (numeric only)
latest_ecg_numbers = []          # ECG numbers exposed to Flutter
latest_rr = {"value": None}      # latest instantaneous RR
resp_rate_history = []           # 1-minute RR averages
last_ecg_time = time.time()

# ======================================================
# NEUROKIT PARAMETERS
# ======================================================
FS = 50                          # ECG sampling rate
WINDOW_SEC = 30
WINDOW_SAMPLES = FS * WINDOW_SEC
HOP_SEC = 10                     # run NK every 10 seconds
INTENSITY_THRESHOLD = 0.05

# ======================================================
# NEUROKIT BACKGROUND WORKER
# ======================================================
def neurokit_worker():
    global latest_rr, resp_rate_history

    rr_temp = []
    minute_start = time.time()

    print("[NK] Background worker started")

    while True:
        time.sleep(HOP_SEC)

        # Need at least 30 seconds of ECG
        if len(ecg_buffer) < WINDOW_SAMPLES:
            continue

        # 30-second sliding window
        segment = np.array(ecg_buffer[-WINDOW_SAMPLES:], dtype=float)
        segment = np.where(segment >= 0, segment, np.nan)
        segment = pd.Series(segment).interpolate().bfill().to_numpy()

        try:
            # ---- ECG-derived respiration ----
            edr = nk.ecg_rsp(segment, sampling_rate=FS)

            # ---- Breathing intensity ----
            rsp_intensity = (
                pd.Series(edr)
                .rolling(int(3 * FS), center=True)
                .std()
                .fillna(0)
            )

            # ---- Respiration rate ----
            rr = np.array(nk.rsp_rate(edr, sampling_rate=FS))
            valid_rr = rr[rsp_intensity >= INTENSITY_THRESHOLD]
            valid_rr = valid_rr[valid_rr > 0]

            if len(valid_rr) > 0:
                rr_val = float(np.mean(valid_rr))
                latest_rr["value"] = rr_val
                rr_temp.append(rr_val)
                print(f"[NK] RR (10 s hop): {rr_val:.2f}")

        except Exception as e:
            print("[NK] Error:", e)
            continue

        # ---- 1-minute average (≈4 samples) ----
        if time.time() - minute_start >= 60:
            if rr_temp:
                avg_rr = float(np.mean(rr_temp))
                resp_rate_history.append(avg_rr)
                print(f"[NK] 1-min RR: {avg_rr:.2f}")

            rr_temp.clear()
            minute_start = time.time()

# ======================================================
# AUTO-CLEAR ECG IF ESP STOPS
# ======================================================
def ecg_auto_clear_loop():
    global ecg_buffer, latest_ecg_numbers, last_ecg_time

    while True:
        time.sleep(30)
        if time.time() - last_ecg_time > 300:  # 5 minutes
            ecg_buffer.clear()
            latest_ecg_numbers.clear()
            print("[AUTO CLEAR] ECG buffers cleared (timeout)")

# ======================================================
# DATA INGESTION (ESP32)
# ======================================================
@app.route("/data", methods=["POST"])
def receive_data():
    global last_ecg_time

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "bad JSON"}), 400

    ecg = data.get("ecg")
    timestamp = data.get("timestamp", time.time())

    # Batch ECG
    if isinstance(ecg, list):
        for v in ecg:
            ecg_buffer.append(v)
            latest_ecg_numbers.append(v)

    # Single ECG sample
    elif isinstance(ecg, (int, float)):
        ecg_buffer.append(ecg)
        latest_ecg_numbers.append(ecg)

    last_ecg_time = time.time()
    return jsonify({"status": "ok"})

# ======================================================
# ECG ENDPOINTS (FLUTTER)
# ======================================================
@app.route("/ecgnumbers", methods=["GET"])
def get_ecg_numbers():
    if not latest_ecg_numbers:
        return jsonify({"numbers": []}), 404
    return jsonify({"numbers": latest_ecg_numbers})

@app.route("/ecg", methods=["GET"])
def get_ecg_buffer():
    return jsonify(ecg_buffer[-500:])

@app.route("/ecgclear", methods=["POST"])
def clear_ecg():
    ecg_buffer.clear()
    latest_ecg_numbers.clear()
    return jsonify({"status": "ecg cleared"})

# ======================================================
# RESPIRATION ENDPOINTS
# ======================================================
@app.route("/resp_rate", methods=["GET"])
def get_resp_rate():
    return jsonify({"resp_rate": latest_rr["value"]})

@app.route("/resp_history", methods=["GET"])
def get_resp_history():
    return jsonify({"resp_history": resp_rate_history})

# ======================================================
# CLEAR EVERYTHING
# ======================================================
@app.route("/clear_all", methods=["POST"])
def clear_all():
    ecg_buffer.clear()
    latest_ecg_numbers.clear()
    resp_rate_history.clear()
    latest_rr["value"] = None
    return jsonify({"status": "all cleared"})

# ======================================================
# TEST
# ======================================================
@app.route("/test")
def test():
    return "Server running"

# ======================================================
# START SERVER (AZURE VPS SAFE)
# ======================================================
if __name__ == "__main__":
    threading.Thread(target=neurokit_worker, daemon=True).start()
    threading.Thread(target=ecg_auto_clear_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8000)
