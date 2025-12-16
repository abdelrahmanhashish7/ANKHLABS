from flask import Flask, request, jsonify
import threading
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
from collections import deque

app = Flask(__name__)

# ======================================================
# LOGGING (TERMINAL + API)
# ======================================================
LOG_BUFFER_SIZE = 500
server_logs = deque(maxlen=LOG_BUFFER_SIZE)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    server_logs.append(line)

# ======================================================
# GLOBAL SHARED STATE (SINGLE PROCESS)
# ======================================================
ecg_buffer = []                  # sliding ECG buffer
latest_ecg_numbers = []          # exposed to Flutter
latest_rr_1min = None            # latest 1-minute RR
resp_rate_history = []           # history of 1-minute RR
last_ecg_time = time.time()

# ======================================================
# BUFFER CONFIG
# ======================================================
BATCH_SIZE = 500
MAX_ECG_BUFFER = 4000

# ======================================================
# NEUROKIT PARAMETERS
# ======================================================
FS = 50
WINDOW_SEC = 30
WINDOW_SAMPLES = FS * WINDOW_SEC
HOP_SEC = 10
INTENSITY_THRESHOLD = 0.01

# ======================================================
# NEUROKIT BACKGROUND WORKER
# ======================================================
def neurokit_worker():
    global latest_rr_1min

    rr_temp = []
    minute_start = time.time()

    log("[NK] Background worker started")

    while True:
        time.sleep(HOP_SEC)

        buf_len = len(ecg_buffer)
        log(f"[NK] ECG buffer size: {buf_len}")

        if buf_len < WINDOW_SAMPLES:
            continue

        segment = np.array(ecg_buffer[-WINDOW_SAMPLES:], dtype=float)
        segment = np.where(segment >= 0, segment, np.nan)
        segment = pd.Series(segment).interpolate().bfill().to_numpy()

        if np.std(segment) < 1e-3:
            log("[NK] ECG too flat → skipping RR")
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
                rr_temp.append(rr_val)
                log(f"[NK] RR (10 s hop): {rr_val:.2f}")
            else:
                log("[NK] RR invalid → skipped")

        except Exception as e:
            log(f"[NK] Error: {e}")
            continue

        # ===== 1-MINUTE AVERAGE RR =====
        if time.time() - minute_start >= 60:
            if rr_temp:
                latest_rr_1min = float(np.mean(rr_temp))
                resp_rate_history.append(latest_rr_1min)
                log(f"[NK] 1-min RR: {latest_rr_1min:.2f}")
            else:
                log("[NK] No valid RR this minute")

            rr_temp.clear()
            minute_start = time.time()

# ======================================================
# AUTO-CLEAR ECG IF ESP STOPS
# ======================================================
def ecg_auto_clear_loop():
    global ecg_buffer, latest_ecg_numbers

    while True:
        time.sleep(30)
        if time.time() - last_ecg_time > 300:
            ecg_buffer.clear()
            latest_ecg_numbers.clear()
            log("[AUTO CLEAR] ECG buffers cleared (timeout)")

# ======================================================
# DATA INGESTION (ESP32)
# ======================================================
@app.route("/data", methods=["POST"])
def receive_data():
    global last_ecg_time

    data = request.get_json(silent=True)
    if not data:
        log("[ESP] Bad JSON received")
        return jsonify({"status": "error"}), 400

    ecg = data.get("ecg")

    if isinstance(ecg, list):
        ecg_buffer.extend(ecg)
        latest_ecg_numbers.extend(ecg)

        # ---- SLIDING WINDOW (DROP FIRST 500 IF > 4000) ----
        if len(ecg_buffer) > MAX_ECG_BUFFER:
            ecg_buffer[:] = ecg_buffer[BATCH_SIZE:]
        if len(latest_ecg_numbers) > MAX_ECG_BUFFER:
            latest_ecg_numbers[:] = latest_ecg_numbers[BATCH_SIZE:]

        log(f"[ESP] ECG batch received: {len(ecg)} | NK buffer: {len(ecg_buffer)}")

    elif isinstance(ecg, (int, float)):
        ecg_buffer.append(ecg)
        latest_ecg_numbers.append(ecg)

    last_ecg_time = time.time()
    return jsonify({"status": "ok"})

# ======================================================
# API ENDPOINTS
# ======================================================
@app.route("/resp_rate", methods=["GET"])
def get_resp_rate():
    return jsonify({"resp_rate": latest_rr_1min})

@app.route("/resp_history", methods=["GET"])
def get_resp_history():
    return jsonify({"resp_history": resp_rate_history})

@app.route("/ecgnumbers", methods=["GET"])
def get_ecg_numbers():
    return jsonify({"numbers": latest_ecg_numbers})

@app.route("/logs", methods=["GET"])
def get_logs():
    return jsonify({"logs": list(server_logs)})

@app.route("/clear_all", methods=["POST"])
def clear_all():
    global latest_rr_1min
    ecg_buffer.clear()
    latest_ecg_numbers.clear()
    resp_rate_history.clear()
    latest_rr_1min = None
    log("[API] All buffers cleared")
    return jsonify({"status": "cleared"})

@app.route("/test")
def test():
    return "Server running"

# ======================================================
# START SERVER (SINGLE PROCESS — IMPORTANT)
# ======================================================
if __name__ == "__main__":
    threading.Thread(target=neurokit_worker, daemon=True).start()
    threading.Thread(target=ecg_auto_clear_loop, daemon=True).start()

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
        use_reloader=False,
        threaded=False
    )
