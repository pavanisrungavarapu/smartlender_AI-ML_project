# Smart Lender Project Workflow

## Overview

Smart Lender is developed through a structured machine learning workflow. The project includes data collection, data analysis, preprocessing, model development, evaluation, and Flask web application deployment.

## Epic 1: Data Collection and Architecture Design

### Story 1: Dataset Collection

The loan approval dataset is collected and stored in the project directory. The dataset contains applicant information such as gender, marital status, education, income, loan amount, loan term, credit history, property area, and loan status.

Dataset location:

```text
data/loan_prediction.csv
```

### Story 2: Application Architecture Design

The application architecture defines how data moves through the system.

Workflow:

```text
Applicant Data Input
        ↓
Data Preprocessing
        ↓
Machine Learning Model
        ↓
Prediction Result
        ↓
Flask Web Application Output
```

The system consists of:

- Dataset storage
- Data preprocessing module
- Machine learning model training
- Saved model file
- Flask backend
- HTML/CSS frontend
- Prediction result page

## Epic 2: Visualizing and Analyzing the Data

### Story 1: Import And Read Dataset

The dataset is imported using Pandas.

```python
import pandas as pd

df = pd.read_csv("data/loan_prediction.csv")
print(df.head())
```

### Story 2: Univariate Analysis

Univariate analysis is performed to understand individual features such as applicant income, loan amount, education, and credit history.

Examples:

- Count of approved and rejected loans
- Distribution of applicant income
- Distribution of loan amount
- Frequency of credit history values

### Story 3: Bivariate Analysis

Bivariate analysis is used to compare two variables and identify relationships. Count plots are useful for examining how categorical variable pairs differ across applicant groups.

Examples:

- Gender vs married status: segmenting the gender column against the married column reveals patterns in loan applications based on demographic combinations.
- Education vs self-employment status: shows the tendency for educated applicants to be employed.
- Property area vs loan amount term: reveals that loan amount term distributions vary based on the applicant’s property area.
- Credit history vs loan status: helps identify how past credit behavior relates to loan approval.

### Story 4: Multivariate Analysis

Multivariate analysis is used to study relationships among multiple variables together. This project uses the seaborn `swarmplot` function to compare features like property area, education, and loan amount term in a single visualization.

Examples:

- Income, credit history, and loan status
- Education, employment status, and loan approval
- Loan amount, loan term, and property area

## Epic 3: Data Pre-Processing

### Story 1: Handle Missing Values And Duplicates

The dataset is checked for missing values, duplicates, and inconsistent records. Missing values are handled using imputation techniques.

Steps:

- Check missing values
- Remove duplicate rows
- Fill missing numerical values using mean
- Fill missing categorical values using most frequent value (mode)

### Story 2: Balance Dataset

The target column `Loan_Status` is checked to ensure that approved and rejected classes are fairly represented. If the dataset is imbalanced, balancing techniques can be applied.

### Story 3: Scale And Normalize Features

Numerical features such as applicant income, coapplicant income, loan amount, and loan term are scaled using standard scaling. This improves model performance, especially for KNN and other distance-based models.

- Scaling is applied only to the input features (X), not to the target variable `Loan_Status`.
- The preprocessing pipeline transforms `X` into a numeric array before training, while `y` remains unchanged.

### Story 4: Split Dataset

The dataset is split into input features (`X`) and the target variable (`y`) before training.

- `X` contains all feature columns except the target column `Loan_Status`
- `y` contains only the target values for loan approval status

The `train_test_split()` call uses:

- `test_size=0.2` to reserve 20% of the dataset for testing
- `random_state=42` for reproducible splits
- `stratify=y` to preserve the class balance in both train and test sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

## Epic 4: Model Building

### Story 1: Decision Tree Model

A Decision Tree classifier is trained and evaluated on the prepared dataset.

### Story 2: Random Forest Model

A Random Forest classifier is built to improve prediction accuracy using multiple decision trees.

### Story 3: K-Nearest Neighbors Model

A KNN classifier is trained and evaluated after feature scaling.

### Story 4: XGBoost Model

An XGBoost model is trained and compared with other classification algorithms. If XGBoost is unavailable in the local environment, a Gradient Boosting fallback model is used.

### Story 5: Model Evaluation And Saving

All models are evaluated using training accuracy and testing accuracy. The best-performing model is saved for deployment.

Model output file:

```text
models/smart_lender_model.joblib
```

Evaluation output file:

```text
models/metrics.json
```

## Epic 5: Application Building

### Story 1: HTML Page Design

HTML templates are created for user input and result display.

Files:

```text
templates/index.html
templates/result.html
static/css/styles.css
```

### Story 2: Flask Application Development

The Flask application loads the trained model, accepts applicant details from the form, processes the data, and returns a loan approval prediction.

Main file:

```text
app.py
```

### Story 3: Run, Test, And Validate Application

The application is tested locally by running:

```bash
python train_model.py
python app.py
```

Then the app is opened in the browser:

```text
http://127.0.0.1:5000
```

Expected output:

```text
Loan Approved
```

or

```text
Loan Rejected
```

## Final Workflow Summary

The Smart Lender workflow follows the complete machine learning lifecycle:

```text
Dataset Collection
-> Data Analysis
-> Data Preprocessing
-> Model Training
-> Model Evaluation
-> Model Saving
-> Flask Integration
-> Web Application Prediction
```
