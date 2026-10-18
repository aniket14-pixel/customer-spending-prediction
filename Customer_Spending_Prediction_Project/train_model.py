"""
Train and compare regression models for Customer Spending Prediction.
Run from the project root: python train_model.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customer_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
df = df.drop_duplicates().reset_index(drop=True)

if "spending" not in df.columns:
    raise ValueError("Dataset must contain a 'spending' target column.")

# Name is excluded to avoid learning from a customer identifier.
feature_cols = [c for c in df.columns if c not in ["spending", "name"]]
X, y = df[feature_cols], pd.to_numeric(df["spending"], errors="coerce")
valid = y.notna()
X, y = X.loc[valid], y.loc[valid]
numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=42),
}
rows, fitted = [], {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                 "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
    fitted[name] = pipe
metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
best_name = metrics.iloc[0]["Model"]
joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
metrics.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
with open(REPORT_DIR / "project_summary.json", "w") as f:
    json.dump({"best_model": best_name, "best_mae": float(metrics.iloc[0]["MAE"]),
               "best_rmse": float(metrics.iloc[0]["RMSE"]), "best_r2": float(metrics.iloc[0]["R2"]),
               "rows_after_cleaning": int(len(df))}, f, indent=2)
print(metrics.to_string(index=False))
print(f"\nBest model: {best_name}")
print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")
