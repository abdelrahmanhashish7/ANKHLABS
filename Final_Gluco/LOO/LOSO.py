import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("ALLNEW.csv")
features = ["ratio", "ac", "dc", "PI_feature", "slope"]
target = "glucose"

X = df[features].values
y = df[target].values
n_samples = len(df)

print("\nRunning Hyperparameter Tuning...")
param_grid = {
    'n_estimators': [150, 200, 250, 300, 350, 400],
    'max_depth': [None, 5, 7, 10, 12, 15],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ["auto", "sqrt", 0.6, 0.8]
}
base_model = RandomForestRegressor(random_state=42)
tuner = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=25,
    cv=3,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)
tuner.fit(X, y)
best_params = tuner.best_params_
print("Best RF Parameters:", best_params)

# Use best RF for LOO
best_rf = RandomForestRegressor(**best_params, random_state=42)

output_file = "LOO_RF_results.csv"
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_index", "true_value", "predicted_value", "MAE", "RMSE"])

true_values = []
pred_values = []
mae_list = []
rmse_list = []

print("\nRUNNING RANDOM FOREST LOO \n")

for i in range(n_samples):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)
    X_test = X[i].reshape(1, -1)
    y_test = y[i]
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)[0]

    mae = abs(y_test - y_pred)
    rmse = np.sqrt((y_test - y_pred)**2)
    true_values.append(y_test)
    pred_values.append(y_pred)
    mae_list.append(mae)
    rmse_list.append(rmse)

    with open(output_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([i + 1, y_test, y_pred, mae, rmse])

mean_mae = np.mean(mae_list)
mean_rmse = np.mean(rmse_list)

print("\nFINAL LOO PERFORMANCE ")
print("Mean MAE :", round(mean_mae, 2))
print("Mean RMSE:", round(mean_rmse, 2))

# Scatter
plt.figure(figsize=(7, 6))
plt.scatter(true_values, pred_values, alpha=0.7)
plt.plot([min(true_values), max(true_values)],
         [min(true_values), max(true_values)],
         'r--', label="Ideal")
plt.xlabel("True Glucose")
plt.ylabel("Predicted Glucose")
plt.title("Scatter Plot: True vs Predicted (LOO RF)")
plt.legend()
plt.grid(True)
plt.savefig("scatter_true_vs_pred.png", dpi=300)
plt.show()

# Histogram
plt.figure(figsize=(7, 6))
plt.hist(mae_list, bins=15, edgecolor='black')
plt.xlabel("Absolute Error (MAE)")
plt.ylabel("Count")
plt.title("Error Distribution Histogram")
plt.grid(True)
plt.savefig("mae_distribution.png", dpi=300)
plt.show()

# Error per sample
plt.figure(figsize=(10, 5))
plt.plot(mae_list, marker='o')
plt.xlabel("Sample Index")
plt.ylabel("Absolute Error (MAE)")
plt.title("Error Across Samples")
plt.grid(True)
plt.savefig("error_per_sample.png", dpi=300)
plt.show()

print("\nSaved graphs:")
print("- scatter_true_vs_pred.png")
print("- mae_distribution.png")
print("- error_per_sample.png")
