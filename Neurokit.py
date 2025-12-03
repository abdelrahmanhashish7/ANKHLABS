import time
import io
import base64
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")          # important: headless backend
import matplotlib.pyplot as plt
import neurokit2 as nk
from threading import Event

# CONFIG: put your public Azure URL here
BASE_URL = "https://YOURAPP.azurewebsites.net"
FLASK_CLEAR  = f"{BASE_URL}/clear"
FLASK_NUMBERS = f"{BASE_URL}/ecgnumbers"
FLASK_RR = f"{BASE_URL}/resp_rate"
FLASK_HISTORY = f"{BASE_URL}/resp_history"
FLASK_PLOT = f"{BASE_URL}/ecg_plot"
FLASK_ECG = f"{BASE_URL}/ecg"

# Sampling/window params
fs = 50
window_sec = 30
window_samples = fs * window_sec

stop_event = Event()

def process_window(segment):
    """Run NeuroKit processing on a 1-window numpy array and return summary results."""
    try:
        # Example: compute EDR and respiration rate
        edr = nk.ecg_rsp(segment, sampling_rate=fs)
        rr = nk.rsp_rate(edr, sampling_rate=fs)
        rr_val = float(np.mean([v for v in rr if v > 0])) if len(rr) else 0.0
        return {"rr": rr_val, "edr": edr.tolist()}
    except Exception as e:
        print("NeuroKit error:", e)
        return {"rr": 0.0, "edr": []}

def make_plot(buffer_segment):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(buffer_segment[-fs*10:] if len(buffer_segment) >= fs*10 else buffer_segment)
    ax.set_title("ECG (last segment)")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def run_processing():
    """Main loop: fetch data from /ecg, process, post results. Non-blocking & headless."""
    raw_buffer = []
    resp_rate_history = []
    minute_start = time.time()

    # optional: clear server buffers before start
    try:
        requests.post(FLASK_CLEAR, timeout=2)
    except Exception:
        pass

    while not stop_event.is_set():
        # 1) fetch latest samples (small timeout)
        try:
            r = requests.get(FLASK_ECG, timeout=1.0)
            if r.status_code == 200:
                data = r.json()
                samples = [d["ecg"] for d in data if d.get("ecg") is not None and d["ecg"] >= 0]
                if samples:
                    raw_buffer.extend(samples)
                    # send raw numbers back for quick UI update (optional)
                    try:
                        requests.post(FLASK_NUMBERS, json={"numbers": samples}, timeout=1)
                    except Exception:
                        pass
        except Exception:
            # timeout or no data available — continue
            pass

        # 2) keep buffer size sane
        if len(raw_buffer) > window_samples * 2:
            raw_buffer = raw_buffer[-window_samples*2:]

        # 3) process if enough data for a full window
        if len(raw_buffer) >= window_samples:
            window = np.array(raw_buffer[-window_samples:])
            # fill/clean
            window = np.where(window >= 0, window, np.nan)
            window = pd.Series(window).interpolate().fillna(method="bfill").to_numpy()

            result = process_window(window)
            rr_val = result.get("rr", 0.0)
            # post respiration rate
            try:
                requests.post(FLASK_RR, json={"resp_rate": rr_val}, timeout=1)
            except Exception:
                pass

            # create and send plot (headless)
            try:
                plot_b64 = make_plot(window)
                requests.post(FLASK_PLOT, json={"plot": plot_b64}, timeout=2)
            except Exception:
                pass

        # 4) one-second sleep (cooperative)
        time.sleep(1.0)
