import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

DATA_PATH = "ALLNEW.csv"
OUTPUT_DIR = "offline_model_distilled"
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

features = ["ratio", "ac", "dc", "PI_feature", "slope"]
target = "glucose"

missing = set(features + [target]) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[features]
y = df[target]

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=6,
    min_samples_leaf=4,
    max_features="sqrt",
    bootstrap=True,
    random_state=42
)
print("\nTraining Random Forest:")
rf.fit(X, y)

y_rf_pred = rf.predict(X)
mae_rf = mean_absolute_error(y, y_rf_pred)
r2_rf = r2_score(y, y_rf_pred)
print(f"Random Forest MAE: {mae_rf:.2f}")
print(f"Random Forest R² : {r2_rf:.4f}")

y_teacher = rf.predict(X) #output random forest

lr = LinearRegression()
lr.fit(X, y_teacher)

y_lr_pred = lr.predict(X)
mae_lr = mean_absolute_error(y_teacher, y_lr_pred)
r2_lr = r2_score(y_teacher, y_lr_pred)

print("\nDISTILLED LINEAR MODEL PERFORMANCE:")
print(f"Distilled MAE vs RF : {mae_lr:.2f}")
print(f"Distilled R² vs RF  : {r2_lr:.4f}")

b0 = float(lr.intercept_)
b1, b2, b3, b4, b5 = map(float, lr.coef_)
coeff_line = f"{b0:.6f},{b1:.6f},{b2:.6f},{b3:.6f},{b4:.6f},{b5:.6f}"
coeff_path = os.path.join(OUTPUT_DIR, "distilled_coefficients.csv")
with open(coeff_path, "w") as f:
    f.write("b0,b1,b2,b3,b4,b5\n")
    f.write(coeff_line + "\n")

joblib.dump(rf, os.path.join(OUTPUT_DIR, "teacher_RF_model.joblib"))
joblib.dump(lr, os.path.join(OUTPUT_DIR, "student_LR_model.joblib"))

print("\nFINAL DISTILLED MODEL SAVED ")
print("Coefficients:")
print(coeff_line)
print("\nSaved to:", coeff_path)
