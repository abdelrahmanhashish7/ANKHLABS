import time
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

# ===== These will be injected from Flask =====
ecg_buffer = None
latest_rr = None
resp_rate_history = None
latest_plot = None

# ===== Parameters =====
fs = 50
window_sec = 30
window_samples = fs * window_sec
hop_sec = 10
intensity_threshold = 0.05

def start_worker(
    _ecg_buffer,
    _latest_rr,
    _resp_rate_history,
    _latest_plot
):
    global ecg_buffer, latest_rr, resp_rate_history, latest_plot

    ecg_buffer = _ecg_buffer
    latest_rr = _latest_rr
    resp_rate_history = _resp_rate_history
    latest_plot = _latest_plot

    resp_rate_all = []
    minute_start = time.time()

    print("[NK] Worker started on Azure")

    while True:
        time.sleep(hop_sec)

        if len(ecg_buffer) < window_samples:
            continue

        segment = np.array(ecg_buffer[-window_samples:])
        segment = np.where(segment >= 0, segment, np.nan)
        segment = pd.Series(segment).interpolate().bfill().to_numpy()

        try:
            edr = nk.ecg_rsp(segment, sampling_rate=fs)

            rsp_intensity = (
                pd.Series(edr)
                .rolling(int(3 * fs), center=True)
                .std()
                .fillna(0)
            )

            rr = np.array(nk.rsp_rate(edr, sampling_rate=fs))
            valid_rr = rr[rsp_intensity >= intensity_threshold]
            valid_rr = valid_rr[valid_rr > 0]

            if len(valid_rr) > 0:
                rr_val = float(np.mean(valid_rr))
                resp_rate_all.append(rr_val)
                latest_rr["value"] = rr_val

        except Exception as e:
            print("[NK] Error:", e)
            continue

        # ---- 1-minute average ----
        if time.time() - minute_start >= 60:
            if resp_rate_all:
                avg = float(np.mean(resp_rate_all))
                resp_rate_history.append(avg)
                print("[NK] 1-min RR:", avg)

            resp_rate_all.clear()
            minute_start = time.time()

        # ---- ECG plot ----
        try:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(segment[-500:])
            ax.set_title("ECG (last 10 s)")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            latest_plot["img"] = base64.b64encode(buf.read()).decode()
            plt.close(fig)

        except Exception:
            pass
