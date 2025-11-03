import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import sys

in_path = Path("../data/heart.csv")
out_clean_path = Path("../data/heart_cleaned.csv")

def run_processing(in_path=in_path, out_clean_path=out_clean_path):
    data = pd.read_csv(in_path)

    print("Original Data:")
    print(data.head())

    print("\nMissing values before cleaning:")
    print(data.isnull().sum())

    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        data[col].fillna(data[col].median(), inplace=True)

    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    categorical_cols = data.select_dtypes(include=['object']).columns
    print("\nCategorical columns:", categorical_cols.tolist())

    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    print("\nAfter encoding:")
    print(data.head())

    if "Cholesterol" in data.columns:
        data["Cholesterol_unscaled"] = data["Cholesterol"].copy()
        print("\nSaved Cholesterol_unscaled (unscaled copy).")

    scaler = StandardScaler()

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    print("\nAfter scaling (note: Cholesterol_unscaled preserved):")
    print(data.head())

    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    print("\nFeature shape:", X.shape)
    print("Target shape:", y.shape)

    out_clean_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_clean_path, index=False)
    print(f"\nCleaned dataset saved successfully to {out_clean_path}")

if __name__ == "__main__":
    args = sys.argv[1:]
    inp = in_path
    outp = out_clean_path
    if "--in" in args:
        i = args.index("--in"); inp = Path(args[i+1])
    if "--out" in args:
        j = args.index("--out"); outp = Path(args[j+1])
    run_processing(inp, outp)
