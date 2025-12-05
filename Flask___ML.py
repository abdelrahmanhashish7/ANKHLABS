from flask import Flask, request
import pandas as pd
import os
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# ================== CONFIG ==================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATA_PATH   = os.path.join(DATA_DIR, "training.csv")
MODEL_PATH  = os.path.join(DATA_DIR, "model.csv")
HISTORY_PATH = os.path.join(DATA_DIR, "model_history.csv")

N_MIN_SAMPLES = 20   # must match ESP

# ================== HELPERS ==================

def load_data():
    if not os.path.isfile(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)

    required = {"ratio", "ac", "dc", "PI_feature", "slope", "glucose"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in training.csv: {missing}")

    df = df.dropna(subset=list(required))
    df = df[(df["glucose"] >= 40) & (df["glucose"] <= 400)]

    return df


def train_and_validate(df):
    if len(df) < N_MIN_SAMPLES:
        return None, None, None

    X = df[["ratio", "ac", "dc", "PI_feature", "slope"]]
    y = df["glucose"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, y_pred))

    b0 = float(model.intercept_)
    b1, b2, b3, b4, b5 = [float(c) for c in model.coef_]

    coeff_line = f"{b0:.6f},{b1:.6f},{b2:.6f},{b3:.6f},{b4:.6f},{b5:.6f}"

    # ✅ 1. OVERWRITE latest deployment model
    with open(MODEL_PATH, "w") as f:
        f.write(coeff_line + "\n")

    # ✅ 2. APPEND model history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hist_row = pd.DataFrame([{
        "timestamp": timestamp,
        "n_samples": len(df),
        "rmse": rmse,
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4": b4,
        "b5": b5
    }])

    file_exists = os.path.isfile(HISTORY_PATH)
    hist_row.to_csv(HISTORY_PATH, mode="a",
                    header=not file_exists,
                    index=False)

    return coeff_line, rmse, len(df)


# ================== API ENDPOINT ==================

@app.route("/api/data", methods=["POST"])
def receive_data():
    data = request.get_json()
    if data is None:
        return "ERROR;NO_JSON", 400

    # ✅ Append new training sample
    df_new = pd.DataFrame([data])
    file_exists = os.path.isfile(DATA_PATH)
    df_new.to_csv(DATA_PATH, mode="a", header=not file_exists, index=False)

    df = load_data()
    if df is None:
        return "COLLECTING;N=0", 200

    n_samples = len(df)

    # ✅ Not enough samples yet
    if n_samples < N_MIN_SAMPLES:
        return f"COLLECTING;N={n_samples}", 200

    # ✅ Train + validate + archive model
    coeff_line, rmse, n_samples = train_and_validate(df)

    response = (
        f"READY;N={n_samples};"
        f"RMSE={rmse:.4f};"
        f"COEFFS={coeff_line}"
    )

    return response, 200


# ================== FETCH LATEST MODEL ==================

@app.route("/latest-model", methods=["GET"])
def latest_model():
    if not os.path.isfile(MODEL_PATH):
        return "NO_MODEL", 404

    with open(MODEL_PATH, "r") as f:
        return f.readline().strip(), 200


# ================== OPTIONAL: GET MODEL HISTORY ==================

@app.route("/model-history", methods=["GET"])
def model_history():
    if not os.path.isfile(HISTORY_PATH):
        return "NO_HISTORY", 404

    df = pd.read_csv(HISTORY_PATH)
    return df.to_json(orient="records"), 200


# ================== RUN SERVER ==================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
