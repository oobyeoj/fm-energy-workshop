import pandas as pd
import numpy as np
import joblib
import json

# Load features and model
df = pd.read_csv("data/features.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

with open("models/feature_order.json") as f:
    feature_cols = json.load(f)

model = joblib.load("models/gbr_model.joblib")

# Reproduce the same test split
cutoff = df["timestamp"].max() - pd.Timedelta(days=14)
test = df[df["timestamp"] > cutoff].reset_index(drop=True)

X_test = test[feature_cols]
y_test = test["electricity_kwh"]

# Model predictions
y_pred = model.predict(X_test)

# Baseline: previous hour's consumption (lag_1h)
y_baseline = test["lag_1h"]

# Metrics
def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

model_mae  = mae(y_test, y_pred)
model_rmse = rmse(y_test, y_pred)
base_mae   = mae(y_test, y_baseline)
base_rmse  = rmse(y_test, y_baseline)

print("=== Evaluation on test set (last 14 days, 336 hours) ===")
print(f"\n{'Metric':<8} {'Model':>10} {'Baseline':>10} {'Improvement':>12}")
print("-" * 44)
print(f"{'MAE':<8} {model_mae:>10.3f} {base_mae:>10.3f} {(base_mae - model_mae):>+12.3f}")
print(f"{'RMSE':<8} {model_rmse:>10.3f} {base_rmse:>10.3f} {(base_rmse - model_rmse):>+12.3f}")

print(f"\nMAE as % of mean actual consumption ({y_test.mean():.2f} kWh):")
print(f"  Model    : {model_mae / y_test.mean() * 100:.1f}%")
print(f"  Baseline : {base_mae / y_test.mean() * 100:.1f}%")

# Save predictions for Step 6
test["predicted_kwh"] = y_pred
test.to_csv("data/test_predictions.csv", index=False)
print("\nSaved data/test_predictions.csv")
