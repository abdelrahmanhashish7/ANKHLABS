from flask import Flask, request, jsonify
import threading
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
from collections import deque
import os

app = Flask(__name__)

# ======================================================
# LOGGING (AZURE SAFE)
# ======================================================
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ======================================================
# PARAMETERS
# ======================================================
FS = 50
MAX_ECG_BUFFER = 4000
WINDOW_SEC = 30
WINDOW_SAMPLES = FS * WINDOW_SEC
HOP_SEC = 10
INTENSITY_THRESHOLD = 0.01

# ======================================================
# GLOBAL STATE (SINGLE WORKER ASSUMED)
# ======================================================
ecg_buffer = deque(maxlen=MAX_ECG_BUFFER)
latest_ecg_numbers = deque(maxlen=MAX_ECG_BUFFER)

latest_rr_1min = None
resp_rate_history = []

latest_glucose = None
glucose_history = []

last_ecg_time = time.time()

# ======================================================
# NEUROKIT BACKGROUND WORKER
# ======================================================
def neurokit_worker():
    global latest_rr_1min

    rr_window = []
    minute_start = time.time()

    log("[NK] Worker started")

    while True:
        time.sleep(HOP_SEC)

        if len(ecg_buffer) < WINDOW_SAMPLES:
            continue

        segment = np.array(list(ecg_buffer)[-WINDOW_SAMPLES:], dtype=float)
        segment = np.where(segment >= 0, segment, np.nan)
        segment = pd.Series(segment).interpolate().bfill().to_numpy()

        if np.std(segment) < 1e-3:
            rr_window.append(0.0)
            log("[NK] ECG flat → RR=0")
            continue

        try:
            edr = nk.ecg_rsp(segment, sampling_rate=FS)

            rsp_intensity = (
                pd.Series(edr)
                .rolling(int(3 * FS), center=True)
                .std()
                .fillna(0)
            )

            rr = np.array(nk.rsp_rate(edr, sampling_rate=FS))
            valid_rr = rr[(rsp_intensity >= INTENSITY_THRESHOLD) & (rr > 0)]

            if len(valid_rr) > 0:
                rr_val = float(np.mean(valid_rr))
                rr_window.append(rr_val)
                log(f"[NK] RR (10s): {rr_val:.2f}")
            else:
                rr_window.append(0.0)
                log("[NK] RR invalid → 0")

        except Exception as e:
            rr_window.append(0.0)
            log(f"[NK] Error → RR=0 | {e}")

        # ---------- 1-MIN AVERAGE ----------
        if time.time() - minute_start >= 60:
            if rr_window:
                avg_rr = float(np.mean(rr_window))
                if avg_rr <= 5:
                    avg_rr = 0.0
                    log("[NK] 1-min RR ≤5 → set to 0")

                latest_rr_1min = avg_rr
                resp_rate_history.append(avg_rr)

                log(f"[NK] 1-min RR = {avg_rr:.2f}")

            rr_window.clear()
            minute_start = time.time()

# ======================================================
# ECG AUTO CLEAR (CLOUD SAFETY)
# ======================================================
def ecg_auto_clear():
    global latest_rr_1min
    while True:
        time.sleep(30)
        if time.time() - last_ecg_time > 300:
            ecg_buffer.clear()
            latest_ecg_numbers.clear()
            latest_rr_1min = None
            log("[AUTO] ECG buffers cleared")

# ======================================================
# ECG INGESTION
# ======================================================
@app.route("/data", methods=["POST"])
def receive_ecg():
    global last_ecg_time

    data = request.get_json(silent=True)
    if not data or "ecg" not in data:
        return jsonify({"status": "error"}), 400

    ecg = data["ecg"]

    if isinstance(ecg, list):
        for v in ecg:
            ecg_buffer.append(v)
            latest_ecg_numbers.append(v)
        log(f"[ESP] ECG batch: {len(ecg)} | buffer={len(ecg_buffer)}")
    else:
        ecg_buffer.append(ecg)
        latest_ecg_numbers.append(ecg)

    last_ecg_time = time.time()
    return jsonify({"status": "ok"})

# ======================================================
# GLUCOSE INGESTION
# ======================================================
@app.route("/glucose", methods=["POST"])
def receive_glucose():
    global latest_glucose

    data = request.get_json(silent=True)
    if not data or "glucose" not in data:
        return jsonify({"status": "error"}), 400

    glucose = float(data["glucose"])
    ts = data.get("timestamp", time.time())

    if glucose < 40 or glucose > 400:
        log(f"[ESP] Glucose ignored: {glucose}")
        return jsonify({"status": "ignored"})

    latest_glucose = {"value": glucose, "timestamp": ts}
    glucose_history.append(latest_glucose)

    log(f"[ESP] Glucose received: {glucose:.1f}")
    return jsonify({"status": "ok"})

# ======================================================
# API ENDPOINTS
# ======================================================
@app.route("/resp_rate")
def get_resp_rate():
    return jsonify({"resp_rate": latest_rr_1min})

@app.route("/resp_history")
def get_resp_history():
    return jsonify({"resp_history": resp_rate_history})

@app.route("/ecgnumbers")
def get_ecg_numbers():
    return jsonify({"numbers": list(latest_ecg_numbers)})

@app.route("/latest_glucose")
def get_latest_glucose():
    return jsonify({"glucose": latest_glucose})

@app.route("/glucose_history")
def get_glucose_history():
    return jsonify({"history": glucose_history})

@app.route("/")
def health():
    return "SERVER ONLINE"

# ======================================================
# THREAD START (IMPORTANT FOR GUNICORN)
# ======================================================
_worker_started = False

@app.before_first_request
def start_background_threads():
    global _worker_started
    if _worker_started:
        return

    threading.Thread(target=neurokit_worker, daemon=True).start()
    threading.Thread(target=ecg_auto_clear, daemon=True).start()

    _worker_started = True
    log("[SYSTEM] Background threads started")
