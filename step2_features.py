import pandas as pd

# Load and sort by time
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Keep only the columns we need
df = df[["timestamp", "electricity_kwh"]]

# Time-based features
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek  # Monday=0

# Lag features
df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)

# Drop the first 168 rows that don't have full history
df = df.dropna().reset_index(drop=True)

print("=== Shape after feature engineering ===")
print(f"{df.shape[0]} rows, {df.shape[1]} columns")

print("\n=== Date range after drop ===")
print(f"From : {df['timestamp'].min()}")
print(f"To   : {df['timestamp'].max()}")

print("\n=== First 3 rows ===")
print(df.head(3).to_string(index=False))

print("\n=== Missing values ===")
print(df.isnull().sum().to_string())

print("\n=== Feature columns ===")
feature_cols = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
print(feature_cols)

# Save for later steps
df.to_csv("data/features.csv", index=False)
print("\nSaved to data/features.csv")
