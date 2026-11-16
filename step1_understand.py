import pandas as pd

# Load
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])

print("=== Shape ===")
print(f"{df.shape[0]} rows, {df.shape[1]} columns")

print("\n=== Columns & dtypes ===")
print(df.dtypes)

print("\n=== First 3 rows ===")
print(df.head(3).to_string(index=False))

print("\n=== Date range ===")
print(f"From : {df['timestamp'].min()}")
print(f"To   : {df['timestamp'].max()}")
total_hours = int((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600) + 1
print(f"Expected hours : {total_hours}")
print(f"Actual rows    : {len(df)}")
print(f"Missing hours  : {total_hours - len(df)}")

print("\n=== electricity_kwh stats ===")
print(df["electricity_kwh"].describe().round(3).to_string())

print("\n=== Missing values ===")
print(df.isnull().sum().to_string())

print("\n=== Duplicate timestamps ===")
print(df["timestamp"].duplicated().sum())

print("\n=== Hour-gap check (should all be 1) ===")
gaps = df["timestamp"].sort_values().diff().dropna()
unusual = gaps[gaps != pd.Timedelta(hours=1)]
if unusual.empty:
    print("All consecutive rows are exactly 1 hour apart — no gaps.")
else:
    print(unusual)
