import neurokit2 as nk
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import time
import io, base64

# ==== YOUR PC IP HERE ====
PC_IP = "biosensors-eqgnhhgmb5b9fbex.francecentral-01.azurewebsites.net"
FLASK_ECG = f"https://{PC_IP}/ecg"
FLASK_PLOT = f"https://{PC_IP}/ecg_plot"
FLASK_NUMBERS = f"https://{PC_IP}/ecgnumbers"
FLASK_RR = f"https://{PC_IP}/resp_rate"
FLASK_HISTORY = f"https://{PC_IP}/resp_history"
FLASK_CLEAR = f"https://{PC_IP}/clear"

# ==== Sampling rate of ESP ====
fs = 50   # 50 samples per second

# ==== Sliding window parameters ====
window_sec = 30            # window length in seconds
window_samples = fs * window_sec
step_sec = 1               # update every 1 second
step_samples = fs * step_sec
threshold = 0.05

# ==== Buffers ====
raw_buffer = []        # holds all incoming ECG
edr_all = []           # holds all EDR samples
resp_rate_all = []     # temporary respiration rates
resp_rate_history = [] # last 1-min averages
minute_start = time.time()

# Initialize rsp_intensity globally to prevent undefined errors
rsp_intensity = pd.Series(dtype=float)

# ==== Plotting ====
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# ============================
# FUNCTION: fetch ECG samples
# ============================
def fetch_ecg():
    """Gets all ECG samples from Flask buffer."""
    try:
        r = requests.get(FLASK_ECG, timeout=0.5)
        if r.status_code == 200:
            data = r.json()
            # Filter invalid ECG samples (ECG=-1 indicates invalid)
            return [d["ecg"] for d in data if d["ecg"] >= 0]
        return []
    except Exception:
        return []

# ============================
# UPDATE FUNCTION (1 sec)
# ============================
def update(frame):
    global raw_buffer, edr_all, resp_rate_all, minute_start, resp_rate_history, rsp_intensity

    # --- 0) Ensure rsp_intensity exists even if buffer is small ---
    if len(raw_buffer) == 0:
        rsp_intensity = pd.Series([0])

    # --- 1) Fetch new ECG samples ---
    samples = fetch_ecg()
    if samples:
        raw_buffer.extend(samples)
        try:
            requests.post(FLASK_NUMBERS, json={"numbers": samples})
        except Exception as e:
            print("Could not send ECG numbers:", e)
    else:
        print("No new ECG yet...")
        return

    # --- Limit buffer size ---
    if len(raw_buffer) > window_samples*2:
        raw_buffer = raw_buffer[-window_samples*2:]

    # --- 2) Process when enough samples for window ---
    if len(raw_buffer) >= window_samples:
        segment = np.array(raw_buffer[-window_samples:])
        # Replace invalid values with NaN, interpolate
        segment = np.where(segment >= 0, segment, np.nan)
        segment = pd.Series(segment).interpolate().fillna(method="bfill").to_numpy()

        try:
            # --- Compute EDR ---
            edr = nk.ecg_rsp(segment, sampling_rate=fs)
            edr_all.extend(edr.tolist())

            # --- Rolling Breathing Intensity ---
            window_intensity_sec = 3
            samples_intensity = int(window_intensity_sec * fs)
            edr_series = pd.Series(edr)
            rsp_intensity = edr_series.rolling(samples_intensity, center=True).std().fillna(0)

            # --- Respiration rate calculation ---
            rr = nk.rsp_rate(edr, sampling_rate=fs)
            rr = np.array(rr)

            # --- Filter RR using intensity threshold ---
            
            filtered_rr = np.where(rsp_intensity >= threshold, rr, 0)

            # --- Append valid respiration rates ---
            valid_rr = filtered_rr[filtered_rr > 0]
            if len(valid_rr) > 0:
                rr_val = float(np.mean(valid_rr))
                resp_rate_all.append(rr_val)
                print(f"Filtered Respiration Rate (instant): {rr_val:.2f} bpm")
            else:
                print("Low intensity breathing detected → rate = 0")

        except Exception as e:
            print("Processing error:", e)

    # --- 3) One-minute average respiration rate ---
    elapsed = time.time() - minute_start
    if elapsed >= 60:
        if resp_rate_all:
            avg_rr = np.mean(resp_rate_all)
            resp_rate_history.append(avg_rr)
            print(f"Average Respiration Rate (last 1 min): {avg_rr:.2f} bpm")
            try:
                requests.post(FLASK_RR, json={"resp_rate": avg_rr})
            except Exception as e:
                print("Could not send respiration rate:", e)
            try:
                requests.post(FLASK_HISTORY, json={"resp_history": resp_rate_history})
            except Exception as e:
                print("Could not send respiration history:", e)
        else:
            print("No respiration data for this minute.")

        # --- Clear buffers for next minute ---
        resp_rate_all = []
        edr_all = []
        raw_buffer = []
        minute_start = time.time()
        time.sleep(0.1)

    # ========================
    # 4) PLOTTING
    # ========================
    axs[0].cla()
    if len(raw_buffer) >= fs*10:
        axs[0].plot(raw_buffer[-fs*10:])
    else:
        axs[0].plot(raw_buffer)
    axs[0].set_title("ECG (last 10 seconds)")

    # --- Plot EDR + intensity ---
    axs[1].cla()
    if len(edr_all) >= window_samples:
        axs[1].plot(edr_all[-window_samples:], label="EDR")
        axs[1].plot(rsp_intensity[-window_samples:], label="Breathing Intensity")
        axs[1].axhline(y=threshold, linestyle="--", color="red", label="Threshold")
    else:
        axs[1].plot(edr_all, label="EDR")
        axs[1].plot(rsp_intensity, label="Breathing Intensity")
        axs[1].axhline(y=threshold, linestyle="--", color="red", label="Threshold")
    axs[1].set_title("EDR + Breathing Intensity")
    axs[1].legend()

    # --- Send plot to Flask as base64 ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    try:
        requests.post(FLASK_PLOT, json={"plot": img_b64})
    except Exception as e:
        print("Could not send ECG plot:", e)

# ============================
# INIT: Clear Flask buffers
# ============================
try:
    requests.post(FLASK_CLEAR)
    print("Flask buffer cleared before start.")
except Exception:
    print("Could not clear Flask buffer (endpoint missing?).")

# ============================
# Start animation loop
# ============================
raw_buffer = []
edr_all = []
resp_rate_all = []
resp_rate_history = []
minute_start = time.time()
rsp_intensity = pd.Series([0])

def run_processing():
    while True:
        update(None)
        time.sleep(1)



