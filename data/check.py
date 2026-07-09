import pandas as pd

# Load the dataset
df = pd.read_csv("data/loan_prediction.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())