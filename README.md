# Machine Learning Data Preprocessing using Python

## Overview
This project demonstrates a practical data-preprocessing workflow using Python. The work covers data loading, inspection, cleaning, missing-value handling, duplicate removal, feature selection, categorical encoding, normalization, and basic exploratory data analysis (EDA).

## Dataset
`data/student_performance.csv` contains a small sample student-performance dataset with numerical and categorical variables. A few missing values and a duplicate row are intentionally included so the cleaning steps can be demonstrated.

### Columns
- `Student_ID` – identifier
- `Age` – student age
- `Gender` – categorical feature
- `Attendance` – attendance percentage
- `Study_Hours` – average study hours
- `Department` – academic department
- `Previous_Score` – previous academic score
- `Final_Score` – final score / target

## Preprocessing Steps
1. Load the CSV file with Pandas.
2. Inspect structure, data types, statistics, and missing values.
3. Remove duplicate records.
4. Fill numerical missing values with the median.
5. Fill categorical missing values with the mode.
6. Remove `Student_ID` from the feature set because it is only an identifier.
7. Apply one-hot encoding to categorical features.
8. Apply Min-Max normalization to selected numerical features.
9. Perform basic EDA with Pandas and NumPy.
10. Save the cleaned and preprocessed outputs.

## Run the Project

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python src/preprocess.py
```

The processed files will be created in the `outputs/` folder.

## Project Structure
```text
ml-data-preprocessing/
├── data/
│   └── student_performance.csv
├── src/
│   └── preprocess.py
├── notebooks/
│   └── preprocessing_walkthrough.md
├── outputs/
│   └── (generated after running the script)
├── README.md
└── requirements.txt
```

## Deliverable
The final output is a cleaned dataset and a numerical feature matrix that can be used as the starting point for a machine-learning model.

## Note
The dataset is a small educational sample created for demonstrating the preprocessing workflow. The same steps can be adapted to larger real-world datasets.
