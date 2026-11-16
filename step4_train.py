import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import json
import os

# Load the prepared features
df = pd.read_csv("data/features.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

feature_cols = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target_col = "electricity_kwh"

# Reproduce the same split as Step 3
cutoff = df["timestamp"].max() - pd.Timedelta(days=14)
train = df[df["timestamp"] <= cutoff].reset_index(drop=True)

X_train = train[feature_cols]
y_train = train[target_col]

# Train
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print("=== Training complete ===")
print(f"Model            : {model.__class__.__name__}")
print(f"Trees built      : {model.n_estimators_}")
print(f"Train rows used  : {len(X_train)}")

print("\n=== Feature importances ===")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in importances.items():
    bar = "█" * int(score * 50)
    print(f"  {feat:<12} {score:.4f}  {bar}")

# Save model and feature order
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gbr_model.joblib")
with open("models/feature_order.json", "w") as f:
    json.dump(feature_cols, f)

print("\nSaved models/gbr_model.joblib")
print("Saved models/feature_order.json")
