import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv(r"C:\Users\Shaik Tahseen\OneDrive\Desktop\hdc\data\heart.csv")

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

scaler = StandardScaler()

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

print("\nAfter scaling:")
print(data.head())

X = data.drop("HeartDisease", axis=1)   
y = data["HeartDisease"]                

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

data.to_csv("../data/heart_cleaned.csv", index=False)
print("\nCleaned dataset saved successfully!")

