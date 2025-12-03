# NeuroKit.py (modified for cloud)
import matplotlib
# Use headless backend so matplotlib doesn't try to open a GUI on the server
matplotlib.use('Agg')

import neurokit2 as nk
import numpy as np
import requests
import matplotlib.pyplot as plt
import pandas as pd
import time
import io, base64

# ==== YOUR AZURE HOSTNAME HERE (no protocol prefix in PC_IP OR include full domain, we'll use below) ====
PC_HOST = "biosensors-eqgnhhgmb5b9fbex.francecentral-01.azurewebsites.net"
BASE_URL = f"https://{PC_HOST}"
FLASK_ECG = f"{BASE_URL}/ecg"
FLASK_PLOT = f"{BASE_URL}/ecg_plot"
FLASK_NUMBERS = f"{BASE_URL}/ecgnumbers"
FLASK_RR = f"{BASE_URL}/resp_rate"
FLASK_HISTORY = f"{BASE_URL}/resp_history"
FLASK_CLEAR = f"{BASE_URL}/clear"

# ==== Sampling rate of ESP ====
fs = 50
window_sec = 30
window_samples = fs * window_sec
step_sec = 1
threshold = 0.05

# Buffers / state
raw_buffer = []
edr_all = []
resp_rate_all = []
resp_rate_history = []
minute_start = time.time()
rsp_intensity = pd.Series([0])

# Create the figure once (Agg backend will allow saving) — ok to create here now
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

def fetch_ecg():
    try:
        r = requests.get(FLASK_ECG, timeout=1.0)
        if r.status_code == 200:
            data = r.json()
            return [d["ecg"] for d in data if isinstance(d, dict) and d.get("ecg", -1) >= 0]
        return []
    except Exception:
        return []

def update_once():
    global raw_buffer, edr_all, resp_rate_all, minute_start, resp_rate_history, rsp_intensity

    samples = fetch_ecg()
    if samples:
        raw_buffer.extend(samples)
        try:
            requests.post(FLASK_NUMBERS, json={"numbers": samples}, timeout=1.0)
        except Exception as e:
            print("Could not send ECG numbers:", e)
    else:
        # no data this tick
        return

    if len(raw_buffer) > window_samples * 2:
        raw_buffer = raw_buffer[-window_samples*2:]

    if len(raw_buffer) >= window_samples:
        segment = np.array(raw_buffer[-window_samples:])
        segment = np.where(segment >= 0, segment, np.nan)
        segment = pd.Series(segment).interpolate().fillna(method="bfill").to_numpy()
        try:
            edr = nk.ecg_rsp(segment, sampling_rate=fs)
            edr_all.extend(edr.tolist())

            window_intensity_sec = 3
            samples_intensity = int(window_intensity_sec * fs)
            edr_series = pd.Series(edr)
            rsp_intensity = edr_series.rolling(samples_intensity, center=True).std().fillna(0)

            rr = nk.rsp_rate(edr, sampling_rate=fs)
            rr = np.array(rr)
            filtered_rr = np.where(rsp_intensity >= threshold, rr, 0)
            valid_rr = filtered_rr[filtered_rr > 0]
            if len(valid_rr) > 0:
                rr_val = float(np.mean(valid_rr))
                resp_rate_all.append(rr_val)
                print(f"Filtered Respiration Rate (instant): {rr_val:.2f} bpm")
            else:
                print("Low intensity breathing detected → rate = 0")
        except Exception as e:
            print("Processing error:", e)

    # One-minute averaging
    elapsed = time.time() - minute_start
    if elapsed >= 60:
        if resp_rate_all:
            avg_rr = float(np.mean(resp_rate_all))
            resp_rate_history.append(avg_rr)
            print(f"Average Respiration Rate (last 1 min): {avg_rr:.2f} bpm")
            try:
                requests.post(FLASK_RR, json={"resp_rate": avg_rr}, timeout=1.0)
            except Exception as e:
                print("Could not send respiration rate:", e)
            try:
                requests.post(FLASK_HISTORY, json={"resp_history": resp_rate_history}, timeout=1.0)
            except Exception as e:
                print("Could not send respiration history:", e)
        resp_rate_all.clear()
        edr_all.clear()
        raw_buffer.clear()
        minute_start = time.time()

    # Plot and push to flask
    try:
        axs[0].cla()
        if len(raw_buffer) >= fs*10:
            axs[0].plot(raw_buffer[-fs*10:])
        else:
            axs[0].plot(raw_buffer)
        axs[0].set_title("ECG (last 10 seconds)")

        axs[1].cla()
        if len(edr_all) >= window_samples:
            axs[1].plot(edr_all[-window_samples:], label="EDR")
            axs[1].plot(rsp_intensity[-window_samples:], label="Breathing Intensity")
            axs[1].axhline(y=threshold, linestyle="--", label="Threshold")
        else:
            axs[1].plot(edr_all, label="EDR")
            axs[1].plot(rsp_intensity, label="Breathing Intensity")
        axs[1].set_title("EDR + Breathing Intensity")
        axs[1].legend()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        try:
            requests.post(FLASK_PLOT, json={"plot": img_b64}, timeout=1.0)
        except Exception as e:
            print("Could not send ECG plot:", e)
    except Exception as e:
        print("Plotting error:", e)

# Clear flask buffers at start (non-blocking)
try:
    requests.post(FLASK_CLEAR, timeout=1.0)
    print("Flask buffer cleared before start.")
except Exception:
    pass

def run_processing():
    # This function will be run in a daemon thread by Flask
    print("NeuroKit run_processing started")
    while True:
        update_once()
        time.sleep(step_sec)
