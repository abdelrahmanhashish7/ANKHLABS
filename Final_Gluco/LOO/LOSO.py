import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("ALLNEW.csv")
features = ["ratio", "ac", "dc", "PI_feature", "slope"]
target = "glucose"

X = df[features].values
y = df[target].values
n_samples = len(df)

print("\nRunning Hyperparameter Tuning...")
param_grid = {
    'n_estimators': [150, 200, 250, 300],
    'max_depth': [None, 5, 7, 10],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ["sqrt", "log2", 0.6, None]
}
base_model = RandomForestRegressor(random_state=42)
tuner = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=15,
    cv=3,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)
tuner.fit(X, y)
best_params = tuner.best_params_
print("\nBest RF Parameters:", best_params)

teacher_model = RandomForestRegressor(**best_params, random_state=42)
output_file = "LOO_TEACHER_STUDENT_results.csv"
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_index", "true_value",
                     "teacher_pred", "student_pred", "final_pred",
                     "MAE"])
true_vals = []
teacher_preds_all = []
student_preds_all = []
final_preds_all = []
mae_list = []

print("\nRunning Teacher → Student Meta-Learning (LOO)...\n")

for i in range(n_samples):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)
    X_test = X[i].reshape(1, -1)
    y_test = y[i]

    teacher_model.fit(X_train, y_train)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    oof_teacher = np.zeros(len(X_train))

    for train_idx, val_idx in kf.split(X_train):
        t = RandomForestRegressor(**best_params, random_state=42)
        t.fit(X_train[train_idx], y_train[train_idx])
        oof_teacher[val_idx] = t.predict(X_train[val_idx])

    # Residuals for the student
    residuals = y_train - oof_teacher

    # Train STUDENT on "teacher_pred → residual"
    student = LinearRegression()
    student.fit(oof_teacher.reshape(-1, 1), residuals)

    # Teacher prediction on test
    teacher_pred = teacher_model.predict(X_test)[0]

    # Student correction
    student_pred = student.predict(np.array([[teacher_pred]]))[0]
    final_pred = teacher_pred + student_pred

    # Evaluate
    mae = abs(final_pred - y_test)
    teacher_preds_all.append(teacher_pred)
    student_preds_all.append(student_pred)
    final_preds_all.append(final_pred)
    true_vals.append(y_test)
    mae_list.append(mae)

    with open(output_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([i+1, y_test, teacher_pred, student_pred,
                         final_pred, mae])

print("\nFINAL META-LEARNING LOO PERFORMANCE")
print("Mean MAE :", round(np.mean(mae_list), 3))

plt.figure(figsize=(7, 6))
plt.scatter(true_vals, final_preds_all, alpha=0.7)
plt.plot([min(true_vals), max(true_vals)],
         [min(true_vals), max(true_vals)], 'r--')
plt.xlabel("True Glucose")
plt.ylabel("Predicted Glucose")
plt.title("Teacher → Student Meta-Learning (LOO)")
plt.grid(True)
plt.savefig("scatter.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 6))
plt.hist(mae_list, bins=15, edgecolor='black')
plt.title("MAE Distribution (Teacher → Student)")
plt.xlabel("MAE")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("mae_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(mae_list, marker='o')
plt.title("MAE per Sample (Teacher → Student)")
plt.xlabel("Sample Index")
plt.ylabel("MAE")
plt.grid(True)
plt.savefig("error_per_sample.png", dpi=300)
plt.close()

print("\nSaved graphs:")
print("- scatter.png")
print("- mae_distribution.png")
print("- error_per_sample.png")
