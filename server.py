from flask import Flask, request, jsonify
import threading
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
from collections import deque

app = Flask(__name__)

# ======================================================
# LOGGING
# ======================================================
LOG_BUFFER_SIZE = 500
server_logs = deque(maxlen=LOG_BUFFER_SIZE)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    server_logs.append(line)

# ======================================================
# ECG & RR STATE (SINGLE PROCESS)
# ======================================================
FS = 50
BATCH_SIZE = 500
MAX_ECG_BUFFER = 4000
WINDOW_SEC = 30
WINDOW_SAMPLES = FS * WINDOW_SEC
HOP_SEC = 10
INTENSITY_THRESHOLD = 0.01

ecg_buffer = deque(maxlen=MAX_ECG_BUFFER)
latest_ecg_numbers = deque(maxlen=MAX_ECG_BUFFER)

latest_rr_1min = None
resp_rate_history = []

last_ecg_time = time.time()

# ======================================================
# NEUROKIT BACKGROUND WORKER
# ======================================================
def neurokit_worker():
    global latest_rr_1min

    rr_window = []
    minute_start = time.time()

    log("[NK] Background worker started")

    while True:
        time.sleep(HOP_SEC)

        buf_len = len(ecg_buffer)
        log(f"[NK] ECG buffer size: {buf_len}")

        if buf_len < WINDOW_SAMPLES:
            continue

        segment = np.array(list(ecg_buffer)[-WINDOW_SAMPLES:], dtype=float)
        segment = np.where(segment >= 0, segment, np.nan)
        segment = pd.Series(segment).interpolate().bfill().to_numpy()

        if np.std(segment) < 1e-3:
            log("[NK] ECG too flat → skipping")
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
            valid_rr = rr[rsp_intensity >= INTENSITY_THRESHOLD]
            valid_rr = valid_rr[valid_rr > 0]

            rr_val = None
            if len(valid_rr) > 0:
                rr_val = float(np.mean(valid_rr))
            elif len(rr) > 0:
                fallback = np.nanmean(rr)
                if not np.isnan(fallback):
                    rr_val = float(fallback)

            if rr_val is not None:
                rr_window.append(rr_val)
                log(f"[NK] RR (10 s hop): {rr_val:.2f}")
            else:
                   rr_window.append(0.0)
                   log("[NK] RR invalid → counted as 0")
        except Exception as e:
            log(f"[NK] Error: {e}")
            continue

        # ---- 1-minute average ----
        if time.time() - minute_start >= 60:
            if len(rr_window) > 0:
                avg_rr = float(np.mean(rr_window))
                latest_rr_1min = avg_rr
                resp_rate_history.append(avg_rr)
                log(f"[NK] 1-min RR: {avg_rr:.2f}")
            else:
                log("[NK] No valid RR this minute")

            rr_window.clear()
            minute_start = time.time()

# ======================================================
# AUTO-CLEAR IF ESP STOPS
# ======================================================
def ecg_auto_clear_loop():
    global latest_rr_1min
    while True:
        time.sleep(30)
        if time.time() - last_ecg_time > 300:
            ecg_buffer.clear()
            latest_ecg_numbers.clear()
            latest_rr_1min = None
            log("[AUTO CLEAR] ECG buffers cleared (timeout)")

# ======================================================
# DATA INGESTION
# ======================================================
@app.route("/data", methods=["POST"])
def receive_data():
    global last_ecg_time

    data = request.get_json(silent=True)
    if not data or "ecg" not in data:
        log("[ESP] Invalid JSON")
        return jsonify({"status": "error"}), 400

    ecg = data["ecg"]

    if isinstance(ecg, list):
        for v in ecg:
            ecg_buffer.append(v)
            latest_ecg_numbers.append(v)
        log(f"[ESP] ECG batch received: {len(ecg)} | NK buffer: {len(ecg_buffer)}")
    else:
        ecg_buffer.append(ecg)
        latest_ecg_numbers.append(ecg)

    last_ecg_time = time.time()
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

@app.route("/logs")
def get_logs():
    return jsonify({"logs": list(server_logs)})

@app.route("/clear_all", methods=["POST"])
def clear_all():
    ecg_buffer.clear()
    latest_ecg_numbers.clear()
    resp_rate_history.clear()
    global latest_rr_1min
    latest_rr_1min = None
    log("[API] All buffers cleared")
    return jsonify({"status": "cleared"})

@app.route("/")
def health():
    return "Server running"

# ======================================================
# START BACKGROUND THREADS (IMPORTANT)
# ======================================================
threading.Thread(target=neurokit_worker, daemon=True).start()
threading.Thread(target=ecg_auto_clear_loop, daemon=True).start()

# ======================================================
# LOCAL DEV ONLY
# ======================================================
if __name__ == "__main__":
    print("Run with gunicorn in production")

