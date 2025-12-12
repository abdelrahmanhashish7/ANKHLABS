import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

DATA_PATH = "ALLNEW.csv"
OUTPUT_DIR = "offline_model_hybrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
features = ["ratio", "ac", "dc", "PI_feature", "slope"]
target = "glucose"

X = df[features].values
y = df[target].values

teacher = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

print("\nTraining TEACHER (Random Forest)...")
teacher.fit(X, y)
teacher_pred = teacher.predict(X)
mae_teacher = mean_absolute_error(y, teacher_pred)
r2_teacher = r2_score(y, teacher_pred)

print(f"Teacher RF MAE vs True: {mae_teacher:.3f}")
print(f"Teacher RF R² vs True : {r2_teacher:.4f}")

residuals = y - teacher_pred  # target for the student

student = LinearRegression()
student.fit(X, residuals)
student_residual_pred = student.predict(X)
final_pred = teacher_pred + student_residual_pred

mae_final = mean_absolute_error(y, final_pred)
r2_final = r2_score(y, final_pred)

print("\nRESIDUAL HYBRID MODEL PERFORMANCE:")
print(f"Final MAE vs True: {mae_final:.3f}")
print(f"Final R²  vs True: {r2_final:.4f}")

joblib.dump(teacher, os.path.join(OUTPUT_DIR, "teacher_RF_model.joblib"))
joblib.dump(student, os.path.join(OUTPUT_DIR, "student_residual_LR.joblib"))

b0 = float(student.intercept_)
b = student.coef_
coeff_line = ",".join([f"{b0:.6f}"] + [f"{coef:.6f}" for coef in b])
with open(os.path.join(OUTPUT_DIR, "residual_student_coeff.csv"), "w") as f:
    f.write("b0,b1,b2,b3,b4,b5\n")
    f.write(coeff_line + "\n")

print("HYBRID MODEL SAVED SUCCESSFULLY")
print("Residual Student Coefficients:")
print(coeff_line)
