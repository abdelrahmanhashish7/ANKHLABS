import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def _make_plot(ecg_segment, edr, fs, show_seconds=10):
    """Create a two-row plot: ECG (last show_seconds) and EDR (last window). Return base64 PNG."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    # ECG plot (last show_seconds seconds)
    samples_to_show = int(min(len(ecg_segment), fs * show_seconds))
    axs[0].plot(ecg_segment[-samples_to_show:])
    axs[0].set_title(f"ECG (last {show_seconds} s)")
    axs[0].set_xlabel("Samples")
    axs[0].set_ylabel("Amplitude")

    # EDR plot
    axs[1].plot(edr if len(edr) > 0 else [0])
    axs[1].set_title("EDR (estimated respiration from ECG)")
    axs[1].set_xlabel("Samples")
    axs[1].set_ylabel("EDR")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_b64

def process_data(ecg_list, fs=50, window_sec=30):
    """
    Process ECG samples and return a dict:
      {
        "resp_rate": float or None,
        "plot": base64 PNG string,
        "edr": list,
        "rr_series": list (if available),
        "status": "ok" / "error",
        "message": optional message
      }

    Expect ecg_list = list or numpy array of raw ECG numeric samples (>=0 valid).
    """
    try:
        if not ecg_list or len(ecg_list) < 2:
            return {"status": "error", "message": "Not enough ECG samples", "resp_rate": None}

        # Keep only numeric values
        arr = np.array(ecg_list, dtype=float)

        # Replace invalid (<0) with NaN, interpolate
        arr = np.where(arr >= 0, arr, np.nan)
        arr = pd.Series(arr).interpolate().fillna(method="bfill").to_numpy()

        # Use last window_sec seconds if available
        window_samples = int(window_sec * fs)
        if len(arr) >= window_samples:
            segment = arr[-window_samples:]
        else:
            segment = arr

        # Compute EDR (ecg_rsp) and respiration rate
        try:
            edr = nk.ecg_rsp(segment, sampling_rate=fs)
        except Exception:
            # Fallback: try ecg_process then derive respiration
            try:
                proc = nk.ecg_process(segment, sampling_rate=fs)
                # try to locate RSP column if produced
                edr = proc.get("ECG_RSP", np.zeros_like(segment))
            except Exception:
                edr = np.zeros_like(segment)

        # Respiration rate series (may return array-like)
        rr_series = None
        resp_rate = None
        try:
            rr = nk.rsp_rate(edr, sampling_rate=fs)
            # rr may be array-like; remove nan and compute mean
            rr_arr = np.array(rr, dtype=float)
            rr_clean = rr_arr[~np.isnan(rr_arr)]
            if rr_clean.size > 0:
                resp_rate = float(np.mean(rr_clean))
                rr_series = rr_clean.tolist()
            else:
                resp_rate = None
        except Exception:
            resp_rate = None
            rr_series = None

        # Create plot as base64 PNG
        plot_b64 = _make_plot(segment, edr, fs)

        return {
            "status": "ok",
            "resp_rate": resp_rate,
            "plot": plot_b64,
            "edr": edr.tolist() if hasattr(edr, "tolist") else list(edr),
            "rr_series": rr_series
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "resp_rate": None}