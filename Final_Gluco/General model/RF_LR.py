import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "new.csv"
df = pd.read_csv(DATA_PATH)

FEATURES = ["ratio", "ac", "dc", "PI_feature", "slope"]
TARGET = "glucose"
SUBJECT_COL = "subject_id"

df_rel = df.copy()
for f in FEATURES:
    df_rel[f + "_base"] = df_rel.groupby(SUBJECT_COL)[f].transform("mean")
    df_rel[f + "_rel"]  = df_rel[f] - df_rel[f + "_base"]

REL_FEATURES = [f + "_rel" for f in FEATURES]

X = df_rel[REL_FEATURES].values
y = df_rel[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)

rf.fit(X_train_s, y_train)
rf_train_pred = rf.predict(X_train_s)
rf_test_pred  = rf.predict(X_test_s)

print("\n TEACHER: RANDOM FOREST ")
print("TRAIN MAE:", mean_absolute_error(y_train, rf_train_pred))
print("TRAIN R² :", r2_score(y_train, rf_train_pred))
print("TEST  MAE:", mean_absolute_error(y_test, rf_test_pred))
print("TEST  R² :", r2_score(y_test, rf_test_pred))


y_teacher_train = rf_train_pred
y_teacher_test  = rf_test_pred
lin = LinearRegression()
lin.fit(X_train_s, y_teacher_train)
lin_train_pred = lin.predict(X_train_s)
lin_test_pred  = lin.predict(X_test_s)

print("\nSTUDENT vs TEACHER ")
print("TRAIN MAE:", mean_absolute_error(y_teacher_train, lin_train_pred))
print("TEST  MAE:", mean_absolute_error(y_teacher_test, lin_test_pred))

print("\nFINAL STUDENT vs TRUE ")
print("TEST MAE:", mean_absolute_error(y_test, lin_test_pred))
print("TEST R² :", r2_score(y_test, lin_test_pred))

print("\nCALING (ESP) ")
for name, m, s in zip(REL_FEATURES, scaler.mean_, scaler.scale_):
    print(f"{name}_n = ({name} - {m:.6f}) / {s:.6f}")

print("\nFINAL LINEAR EQUATION (ESP) ")
print(f"glucose = {lin.intercept_:.6f}")
for coef, name in zip(lin.coef_, REL_FEATURES):
    print(f"+ ({coef:.6f} * {name}_n)")
