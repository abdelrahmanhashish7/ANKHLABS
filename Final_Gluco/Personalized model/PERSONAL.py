import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "ALLNEW.csv"
OUTPUT_DIR = "per_subject_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

features = ["ratio", "ac", "dc", "PI_feature", "slope"]
target = "glucose"
subjects = df["subject_id"].unique()

print("\nPER SUBJECT MODELS WITH COEFFICIENT EXPORT:\n")
summary = []

for sub in subjects:
    sub_df = df[df["subject_id"] == sub]
    if len(sub_df) < 5:
        print("Subject", sub, ": Not enough data, skipped\n")
        continue

    X = sub_df[features]
    y = sub_df[target]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)
    mean_glucose = np.mean(y)
    accuracy = 1 - (mae / mean_glucose)

    b0 = float(model.intercept_)
    b1, b2, b3, b4, b5 = [float(v) for v in model.coef_]
    coeff_line = "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f" % (b0, b1, b2, b3, b4, b5)
    coeff_path = os.path.join(OUTPUT_DIR, f"subject_{sub}_coeff.csv")

    with open(coeff_path, "w") as f:
        f.write("b0,b1,b2,b3,b4,b5\n")
        f.write(coeff_line + "\n")

    joblib.dump(model, os.path.join(OUTPUT_DIR, f"subject_{sub}_model.joblib"))
    summary.append([sub, len(sub_df), mae, rmse, r2, accuracy])

    print(f"Subject {sub}")
    print("  Samples:", len(sub_df))
    print("  MAE:", round(mae, 2))
    print("  RMSE:", round(rmse, 2))
    print("  R2:", round(r2, 4))
    print("  Accuracy:", f"{accuracy*100:.2f}%")
    print("  Saved:", coeff_path, "\n")

summary_df = pd.DataFrame(
    summary,
    columns=["subject_id", "samples", "MAE", "RMSE", "R2", "Accuracy"]
)
summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
summary_df.to_csv(summary_path, index=False)

print("ALL SUBJECT COEFFICIENTS SAVED")
print(summary_df)
print("\nSummary saved to:", summary_path)
