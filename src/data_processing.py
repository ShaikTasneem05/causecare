from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shutil
import sys

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT.parent / "data" / "heart.csv"
OUT_PATH = ROOT.parent / "data" / "heart_cleaned.csv"
BACKUP_PATH = ROOT.parent / "data" / "heart_cleaned_backup_before_reproc.csv"

def safe_backup(out_path=OUT_PATH, backup_path=BACKUP_PATH):
    if out_path.exists():
        try:
            shutil.copy2(out_path, backup_path)
            print(f"Backed up existing cleaned CSV to: {backup_path}")
        except Exception as e:
            print("Warning: could not backup existing cleaned CSV:", e)

def load_data(path=IN_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {path}")
    return pd.read_csv(path)

def ensure_target_binary(df, target_col="HeartDisease"):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data.")

    vals = pd.unique(df[target_col].dropna())
    if set([0,1]).issuperset(set(vals.tolist())):
        df[target_col] = df[target_col].astype(int)
        return df

    if df[target_col].dtype == object:
        mapping = {}
        for v in vals:
            s = str(v).strip().lower()
            if s in ("yes", "y", "1", "true", "t", "present"):
                mapping[v] = 1
            elif s in ("no", "n", "0", "false", "f", "absent"):
                mapping[v] = 0
        if mapping:
            df[target_col] = df[target_col].map(mapping).astype(int)
            print(f"Mapped string target values to binary using mapping: {mapping}")
            return df

    try:
        median = df[target_col].median()
        print(f"Warning: target '{target_col}' not binary. Converting to binary using median threshold = {median}")
        df[target_col] = (df[target_col] > median).astype(int)
    except Exception as e:
        print("Error converting target to binary:", e)
        raise

    return df

def run_processing(in_path=IN_PATH, out_path=OUT_PATH):
    print("Loading raw data from:", in_path)
    df = load_data(in_path)

    print("\nOriginal Data (top rows):")
    print(df.head())

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    for col in df.select_dtypes(include=['int64','float64']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "", inplace=True)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    df = ensure_target_binary(df, target_col="HeartDisease")
    print("\nHeartDisease value counts:")
    print(df["HeartDisease"].value_counts())

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("\nCategorical columns to encode:", categorical_cols)
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"Encoded {col}: {list(le.classes_)}")

    if "Cholesterol" in df.columns:
        df["Cholesterol_unscaled"] = df["Cholesterol"].copy()
        print("\nSaved Cholesterol_unscaled (original values preserved).")

    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if "HeartDisease" in numeric_cols:
        numeric_cols.remove("HeartDisease")

    if "Cholesterol_unscaled" in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != "Cholesterol_unscaled"]

    print("\nNumeric columns to scale (excludes HeartDisease and Cholesterol_unscaled):", numeric_cols)
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("\nAfter scaling (sample rows):")
    print(df.head())

    print("\nFeature shape and target shape:")
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts())

    safe_backup(out_path, BACKUP_PATH)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nCleaned dataset saved successfully to: {out_path}")

if __name__ == "__main__":
    run_processing()
