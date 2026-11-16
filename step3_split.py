import pandas as pd

# Load the prepared features
df = pd.read_csv("data/features.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

feature_cols = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target_col = "electricity_kwh"

# Hold back the last 14 days (336 hours) as the test set
cutoff = df["timestamp"].max() - pd.Timedelta(days=14)
train = df[df["timestamp"] <= cutoff].reset_index(drop=True)
test  = df[df["timestamp"] >  cutoff].reset_index(drop=True)

X_train = train[feature_cols]
y_train = train[target_col]
X_test  = test[feature_cols]
y_test  = test[target_col]

print("=== Split summary ===")
print(f"Cutoff point   : {cutoff}")
print(f"Train rows     : {len(train)}  ({train['timestamp'].min().date()} to {train['timestamp'].max().date()})")
print(f"Test rows      : {len(test)}   ({test['timestamp'].min().date()} to {test['timestamp'].max().date()})")

print("\n=== Feature matrix shapes ===")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

print("\n=== Target value ranges ===")
print(f"y_train  min={y_train.min():.2f}  max={y_train.max():.2f}  mean={y_train.mean():.2f}")
print(f"y_test   min={y_test.min():.2f}  max={y_test.max():.2f}  mean={y_test.mean():.2f}")
