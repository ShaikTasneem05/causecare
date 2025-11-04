from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "heart_cleaned.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"].astype(int)   
    return X, y

def train_and_save():
    X, y = load_data()
    X_proc = X.copy()
    for c in X_proc.columns:
        if X_proc[c].dtype == 'object':
            X_proc[c] = pd.factorize(X_proc[c])[0]
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(solver="liblinear", max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    y_pred_proba_lr = logreg.predict_proba(X_test_scaled)[:,1]
    y_pred_lr = (y_pred_proba_lr >= 0.5).astype(int)
    print("LogReg Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("LogReg ROC-AUC:", roc_auc_score(y_test, y_pred_proba_lr))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_proba_rf = rf.predict_proba(X_test)[:,1]
    y_pred_rf = (y_pred_proba_rf >= 0.5).astype(int)
    print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("RF ROC-AUC:", roc_auc_score(y_test, y_pred_proba_rf))

    joblib.dump(logreg, MODELS_DIR / "logreg.joblib")
    joblib.dump(rf, MODELS_DIR / "rf.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(X_proc.columns.tolist(), f)

    print("Saved models and artifacts to:", MODELS_DIR)

if __name__ == "__main__":
    train_and_save()
