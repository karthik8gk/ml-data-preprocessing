# Preprocessing Walkthrough

## 1. Load the data
The CSV file is loaded into a Pandas DataFrame.

## 2. Inspect the data
The project checks the shape, data types, summary statistics and missing values before making changes.

## 3. Clean the data
Duplicate rows are removed. Missing numerical values are filled using the median, while missing categorical values are filled using the most frequent category (mode).

## 4. Select features
`Student_ID` is excluded because it identifies a student but does not represent a useful predictive feature.

## 5. Encode categories
`Gender` and `Department` are converted into numerical columns using one-hot encoding.

## 6. Normalize numerical data
Age, attendance, study hours and previous score are scaled to a common 0–1 range using Min-Max scaling.

## 7. Explore the data
Pandas descriptive statistics and NumPy calculations are used to understand the numerical variables and final-score distribution.

## 8. Save the results
The script writes the cleaned dataset, preprocessed feature matrix, target column and EDA summary to `outputs/`.
