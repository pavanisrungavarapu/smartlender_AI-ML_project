from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency in some environments
    plt = None
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = DATA_DIR / "loan_prediction.csv"
MODEL_PATH = MODEL_DIR / "smart_lender_model.joblib"
MODEL_PKL_PATH = MODEL_DIR / "smart_lender_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERIC_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

TARGET_COLUMN = "Loan_Status"

if plt is not None:
    plt.style.use("fivethirtyeight")


def read_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(data_path) if data_path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_table(path)


def generate_sample_dataset(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], rows, p=[0.78, 0.22])
    married = rng.choice(["Yes", "No"], rows, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], rows, p=[0.55, 0.18, 0.17, 0.10])
    education = rng.choice(["Graduate", "Not Graduate"], rows, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], rows, p=[0.86, 0.14])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], rows, p=[0.36, 0.39, 0.25])
    credit_history = rng.choice([1.0, 0.0], rows, p=[0.83, 0.17])

    applicant_income = rng.lognormal(mean=8.45, sigma=0.48, size=rows).round().astype(int)
    coapplicant_income = rng.choice([0, 1], rows, p=[0.42, 0.58]) * rng.lognormal(
        mean=7.45, sigma=0.65, size=rows
    ).round().astype(int)
    loan_amount = rng.normal(loc=145, scale=42, size=rows).clip(35, 360).round().astype(int)
    loan_term = rng.choice([120, 180, 240, 300, 360, 480], rows, p=[0.04, 0.07, 0.08, 0.11, 0.66, 0.04])

    total_income = applicant_income + coapplicant_income
    monthly_debt = loan_amount * 1000 / loan_term
    disposable_ratio = total_income / np.maximum(monthly_debt, 1)

    score = (
        1.9 * credit_history
        + 0.45 * (education == "Graduate")
        + 0.35 * (property_area == "Semiurban")
        + 0.18 * (property_area == "Urban")
        + 0.30 * (married == "Yes")
        - 0.50 * (self_employed == "Yes")
        + 0.22 * np.log1p(disposable_ratio)
        - 0.32 * (loan_amount > 210)
        - 0.22 * (dependents == "3+")
        + rng.normal(0, 0.55, rows)
    )

    threshold = np.quantile(score, 0.35)
    loan_status = np.where(score >= threshold, "Y", "N")

    return pd.DataFrame(
        {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )


def load_dataset() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    if DATA_PATH.exists():
        return read_dataset(DATA_PATH)

    df = generate_sample_dataset()
    df.to_csv(DATA_PATH, index=False)
    return read_dataset(DATA_PATH)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print dataset shape, info, and missing-value counts."""
    print(f"Dataset shape: {df.shape}")
    print("\nDataset information:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing pipeline.

    Numeric features use mean imputation followed by standard scaling.
    Categorical features use mode imputation followed by one-hot encoding.
    Scaling is applied only to the input features (X); the target variable (y)
    is not transformed.
    """
    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def preprocess_features(x: pd.DataFrame) -> pd.DataFrame:
    """Transform input features using the preprocessing pipeline.

    Numeric features are imputed and scaled, and categorical features are
    imputed and one-hot encoded. The returned DataFrame contains the transformed
    feature names.
    """
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(x)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=x.index)


def build_training_pipeline(model: object) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model),
        ]
    )


def decisionTree(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Train and evaluate a decision tree classifier on the provided split data."""
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    pipeline = build_training_pipeline(model)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Rejected", "Approved"],
        output_dict=True,
    )
    matrix = confusion_matrix(y_test, predictions)

    return {
        "model": "Decision Tree",
        "pipeline": pipeline,
        "predictions": predictions,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def randomForest(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Train and evaluate a random forest classifier on the provided split data."""
    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=9,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=1,
    )
    pipeline = build_training_pipeline(model)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Rejected", "Approved"],
        output_dict=True,
    )
    matrix = confusion_matrix(y_test, predictions)

    return {
        "model": "Random Forest",
        "pipeline": pipeline,
        "predictions": predictions,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def xgboost(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Train and evaluate a gradient boosting classifier on the provided split data."""
    model = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.06,
        max_depth=3,
        random_state=42,
    )
    pipeline = build_training_pipeline(model)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=["Rejected", "Approved"],
        output_dict=True,
    )
    matrix = confusion_matrix(y_test, predictions)

    return {
        "model": "XGBoost",
        "pipeline": pipeline,
        "predictions": predictions,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def get_models() -> dict[str, object]:
    models: dict[str, object] = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            max_depth=9,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=9),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        models["Gradient Boosting Fallback"] = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=3,
            random_state=42,
        )

    models["Logistic Regression"] = LogisticRegression(max_iter=500, random_state=42)
    return models


def validate_dataset(df: pd.DataFrame) -> None:
    missing = set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_text}")


def train() -> dict[str, object]:
    df = load_dataset()
    validate_dataset(df)
    inspect_dataset(df)

    x = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    y = df[TARGET_COLUMN].map({"N": 0, "Y": 1})

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    results = []
    best_pipeline = None
    best_name = ""
    best_test_accuracy = -1.0
    best_cv_mean = 0.0
    best_cv_std = 0.0
    preferred_order = {
        "Decision Tree": 1,
        "Random Forest": 2,
        "KNN": 3,
        "Logistic Regression": 4,
        "Gradient Boosting Fallback": 5,
        "XGBoost": 6,
    }

    for name, model in get_models().items():
        pipeline = build_training_pipeline(model)
        cv_scores = cross_val_score(
            pipeline,
            x,
            y,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        pipeline.fit(x_train, y_train)

        train_predictions = pipeline.predict(x_train)
        test_predictions = pipeline.predict(x_test)
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        results.append(
            {
                "model": name,
                "cross_validation_mean_accuracy": round(cv_mean, 4),
                "cross_validation_std_accuracy": round(cv_std, 4),
                "training_accuracy": round(float(train_accuracy), 4),
                "testing_accuracy": round(float(test_accuracy), 4),
            }
        )

        is_better_score = test_accuracy > best_test_accuracy
        is_preferred_tie = (
            test_accuracy == best_test_accuracy
            and preferred_order.get(name, 0) > preferred_order.get(best_name, 0)
        )
        if is_better_score or is_preferred_tie:
            best_test_accuracy = test_accuracy
            best_name = name
            best_pipeline = pipeline
            best_cv_mean = cv_mean
            best_cv_std = cv_std

    if best_pipeline is None:
        raise RuntimeError("No model was trained.")

    MODEL_DIR.mkdir(exist_ok=True)
    model_package = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model_package, MODEL_PATH)
    with open(MODEL_PKL_PATH, "wb") as pkl_file:
        pickle.dump(model_package, pkl_file)

    report_predictions = best_pipeline.predict(x_test)
    metrics = {
        "best_model": best_name,
        "model_path": str(MODEL_PATH),
        "pickle_model_path": str(MODEL_PKL_PATH),
        "dataset_path": str(DATA_PATH),
        "cross_validation_mean_accuracy": round(best_cv_mean, 4),
        "cross_validation_std_accuracy": round(best_cv_std, 4),
        "results": results,
        "classification_report": classification_report(
            y_test,
            report_predictions,
            target_names=["Rejected", "Approved"],
            output_dict=True,
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    training_metrics = train()
    print("Smart Lender model training complete")
    print(f"Dataset: {training_metrics['dataset_path']}")
    print(f"Saved model: {training_metrics['model_path']}")
    print()
    for row in training_metrics["results"]:
        training = row["training_accuracy"] * 100
        testing = row["testing_accuracy"] * 100
        print(f"{row['model']}: training={training:.2f}% testing={testing:.2f}%")
    print()
    print(f"Best model: {training_metrics['best_model']}")
