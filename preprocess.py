import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "student_performance.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load the raw dataset
df = pd.read_csv(DATA_FILE)

print("Original shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 2. Remove duplicate records
df = df.drop_duplicates().copy()

# 3. Handle missing values
numeric_cols = ["Age", "Attendance", "Study_Hours", "Previous_Score"]
categorical_cols = ["Gender", "Department"]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Feature selection
# Student_ID is an identifier, so it is not used as a model feature.
feature_cols = [
    "Age", "Gender", "Attendance", "Study_Hours",
    "Department", "Previous_Score"
]
X = df[feature_cols].copy()
y = df["Final_Score"].copy()

# 5. One-hot encode categorical variables
X = pd.get_dummies(X, columns=["Gender", "Department"], drop_first=True)

# 6. Normalize numerical features using Min-Max scaling
scale_cols = ["Age", "Attendance", "Study_Hours", "Previous_Score"]
scaler = MinMaxScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])

# 7. EDA summaries
summary = df[["Attendance", "Study_Hours", "Previous_Score", "Final_Score"]].describe()
print("\nDescriptive statistics:")
print(summary)

print("\nMean final score:", np.mean(df["Final_Score"]))
print("Median final score:", np.median(df["Final_Score"]))

# 8. Save outputs
df.to_csv(OUTPUT_DIR / "cleaned_student_performance.csv", index=False)
X.to_csv(OUTPUT_DIR / "preprocessed_features.csv", index=False)
y.to_csv(OUTPUT_DIR / "target_final_score.csv", index=False)
summary.to_csv(OUTPUT_DIR / "eda_summary.csv")

print("\nPreprocessing complete.")
print("Cleaned dataset:", df.shape)
print("Feature matrix:", X.shape)
