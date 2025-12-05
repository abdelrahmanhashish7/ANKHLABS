import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ==============================
# CONFIG
# ==============================
DEFAULT_OUTPUT_DIR = "offline_model"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
def load_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"ratio", "ac", "dc", "PI_feature", "slope", "glucose"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Basic cleaning
    df = df.dropna()
    df = df[(df["glucose"] >= 40) & (df["glucose"] <= 400)]

    return df


# ==============================
# TRAIN MODEL
# ==============================
def train_model(df, test_size=0.2):
    X = df[["ratio", "ac", "dc", "PI_feature", "slope"]]
    y = df["glucose"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return model, mae, rmse, r2


# ==============================
# SAVE MODEL
# ==============================
def save_model(model, output_dir):
    b0 = float(model.intercept_)
    b1, b2, b3, b4, b5 = [float(v) for v in model.coef_]

    coeff_line = f"{b0:.6f},{b1:.6f},{b2:.6f},{b3:.6f},{b4:.6f},{b5:.6f}"

    coeff_path = os.path.join(output_dir, "final_model.csv")
    with open(coeff_path, "w") as f:
        f.write(coeff_line + "\n")

    # Save sklearn model as backup
    joblib_path = os.path.join(output_dir, "final_model.joblib")
    joblib.dump(model, joblib_path)

    return coeff_path, joblib_path


# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Offline Glucose Model Trainer")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training.csv from Flask"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Validation split (default 0.2)"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for final model"
    )

    args = parser.parse_args()

    print("\n==============================")
    print("OFFLINE GLUCOSE MODEL TRAINER")
    print("==============================")

    df = load_data(args.data)
    print(f"Loaded {len(df)} samples")

    model, mae, rmse, r2 = train_model(df, test_size=args.test_size)

    print("\n===== FINAL MODEL METRICS =====")
    print(f"MAE  : {mae:.2f} mg/dL")
    print(f"RMSE : {rmse:.2f} mg/dL")
    print(f"R²   : {r2:.4f}")

    os.makedirs(args.output, exist_ok=True)
    coeff_path, joblib_path = save_model(model, args.output)

    print("\n===== MODEL SAVED =====")
    print(f"Coefficients : {coeff_path}")
    print(f"Sklearn Model: {joblib_path}")

    print("\n===== COEFFICIENT ORDER =====")
    print("b0, b1, b2, b3, b4, b5")
    print("Intercept, ratio, ac, dc, PI_feature, slope")

    with open(coeff_path, "r") as f:
        print("\nFinal Model:")
        print(f.read())

    print("==============================")
    print("TRAINING COMPLETE")
    print("==============================\n")


if __name__ == "__main__":
    main()

