import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json

# Load predictions
test = pd.read_csv("data/test_predictions.csv", parse_dates=["timestamp"])
test = test.sort_values("timestamp").reset_index(drop=True)

# Plot actual vs predicted over the two-week test window
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top panel: full two-week view
axes[0].plot(test["timestamp"], test["electricity_kwh"], label="Actual", linewidth=1.2)
axes[0].plot(test["timestamp"], test["predicted_kwh"], label="Predicted", linewidth=1.2, linestyle="--")
axes[0].set_title("Actual vs Predicted — full two-week test window")
axes[0].set_ylabel("kWh")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bottom panel: first 7 days only, easier to read
week = test[test["timestamp"] < test["timestamp"].min() + pd.Timedelta(days=7)]
axes[1].plot(week["timestamp"], week["electricity_kwh"], label="Actual", linewidth=1.2)
axes[1].plot(week["timestamp"], week["predicted_kwh"], label="Predicted", linewidth=1.2, linestyle="--")
axes[1].set_title("Actual vs Predicted — first 7 days of test window")
axes[1].set_ylabel("kWh")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/actual_vs_predicted.png", dpi=120)
print("Saved models/actual_vs_predicted.png")

# Confirm model and feature order files exist
import os
for path in ["models/gbr_model.joblib", "models/feature_order.json"]:
    size = os.path.getsize(path)
    print(f"  {path}  ({size:,} bytes)")

# Spot-check: print 10 example rows
print("\n=== Sample predictions (every 33rd row) ===")
sample = test.iloc[::33][["timestamp", "electricity_kwh", "predicted_kwh"]].copy()
sample["error_kwh"] = (sample["electricity_kwh"] - sample["predicted_kwh"]).round(3)
print(sample.to_string(index=False))
