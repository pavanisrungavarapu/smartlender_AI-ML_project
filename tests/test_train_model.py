import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train_model import read_dataset


class TrainModelTests(unittest.TestCase):
    def test_read_dataset_from_csv(self) -> None:
        data_path = Path("data/loan_prediction.csv")

        df = read_dataset(data_path)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("Loan_Status", df.columns)

    def test_preprocessor_uses_mean_and_mode_imputation(self) -> None:
        from train_model import build_preprocessor

        preprocessor = build_preprocessor()
        transformers = {name: transformer for name, transformer, _ in preprocessor.transformers}
        numeric_imputer = transformers["num"].named_steps["imputer"]
        categorical_imputer = transformers["cat"].named_steps["imputer"]

        self.assertEqual(numeric_imputer.strategy, "mean")
        self.assertEqual(categorical_imputer.strategy, "most_frequent")

    def test_preprocessor_uses_standard_scaler_for_numeric_features(self) -> None:
        from sklearn.preprocessing import StandardScaler
        from train_model import build_preprocessor

        preprocessor = build_preprocessor()
        transformers = {name: transformer for name, transformer, _ in preprocessor.transformers}
        numeric_scaler = transformers["num"].named_steps["scaler"]

        self.assertIsInstance(numeric_scaler, StandardScaler)

    def test_training_pipeline_applies_smote(self) -> None:
        from train_model import SMOTE, build_training_pipeline
        from sklearn.tree import DecisionTreeClassifier

        pipeline = build_training_pipeline(DecisionTreeClassifier())
        self.assertIn("smote", pipeline.named_steps)
        self.assertIsInstance(pipeline.named_steps["smote"], SMOTE)

    def test_preprocess_features_returns_dataframe(self) -> None:
        from train_model import preprocess_features

        data = {
            "Gender": ["Male", "Female"],
            "Married": ["Yes", "No"],
            "Dependents": ["0", "1"],
            "Education": ["Graduate", "Not Graduate"],
            "Self_Employed": ["No", "Yes"],
            "ApplicantIncome": [5000, 3000],
            "CoapplicantIncome": [0, 1500],
            "LoanAmount": [100, 120],
            "Loan_Amount_Term": [360, 360],
            "Credit_History": [1.0, 0.0],
            "Property_Area": ["Urban", "Rural"],
        }
        df = pd.DataFrame(data)

        processed = preprocess_features(df)

        self.assertIsInstance(processed, pd.DataFrame)
        self.assertEqual(processed.shape[0], df.shape[0])

    def test_decision_tree_trains_and_evaluates(self) -> None:
        from train_model import decisionTree

        data = {
            "Gender": ["Male", "Female", "Male", "Female"],
            "Married": ["Yes", "No", "Yes", "No"],
            "Dependents": ["0", "1", "0", "1"],
            "Education": ["Graduate", "Not Graduate", "Graduate", "Not Graduate"],
            "Self_Employed": ["No", "Yes", "No", "Yes"],
            "ApplicantIncome": [5000, 3000, 6000, 2500],
            "CoapplicantIncome": [0, 1500, 500, 1000],
            "LoanAmount": [100, 120, 130, 110],
            "Loan_Amount_Term": [360, 360, 360, 360],
            "Credit_History": [1.0, 0.0, 1.0, 0.0],
            "Property_Area": ["Urban", "Rural", "Semiurban", "Urban"],
            "Loan_Status": ["Y", "N", "Y", "N"],
        }
        df = pd.DataFrame(data)
        x = df[[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ]]
        y = df["Loan_Status"].map({"N": 0, "Y": 1})

        x_train = x.iloc[:2]
        x_test = x.iloc[2:]
        y_train = y.iloc[:2]
        y_test = y.iloc[2:]

        result = decisionTree(x_train, x_test, y_train, y_test)

        self.assertEqual(result["model"], "Decision Tree")
        self.assertIn("confusion_matrix", result)
        self.assertIn("classification_report", result)
        self.assertEqual(len(result["predictions"]), len(x_test))
        self.assertEqual(len(result["confusion_matrix"],), 2)
        self.assertIn("Approved", result["classification_report"])

    def test_train_saves_joblib_and_pickle_and_reports_cross_validation(self) -> None:
        import train_model

        original_model_dir = train_model.MODEL_DIR
        original_data_dir = train_model.DATA_DIR
        original_model_path = train_model.MODEL_PATH
        original_model_pkl_path = train_model.MODEL_PKL_PATH
        original_metrics_path = train_model.METRICS_PATH
        original_data_path = train_model.DATA_PATH

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_model.BASE_DIR = tmp_path
            train_model.DATA_DIR = tmp_path / "data"
            train_model.MODEL_DIR = tmp_path / "models"
            train_model.DATA_PATH = train_model.DATA_DIR / "loan_prediction.csv"
            train_model.MODEL_PATH = train_model.MODEL_DIR / "smart_lender_model.joblib"
            train_model.MODEL_PKL_PATH = train_model.MODEL_DIR / "smart_lender_model.pkl"
            train_model.METRICS_PATH = train_model.MODEL_DIR / "metrics.json"

            metrics = train_model.train()

            self.assertTrue(train_model.MODEL_PATH.exists())
            self.assertTrue(train_model.MODEL_PKL_PATH.exists())
            self.assertTrue(train_model.METRICS_PATH.exists())
            self.assertIn("pickle_model_path", metrics)
            self.assertIn("cross_validation_mean_accuracy", metrics)
            self.assertIn("cross_validation_std_accuracy", metrics)

        train_model.MODEL_DIR = original_model_dir
        train_model.DATA_DIR = original_data_dir
        train_model.MODEL_PATH = original_model_path
        train_model.MODEL_PKL_PATH = original_model_pkl_path
        train_model.METRICS_PATH = original_metrics_path
        train_model.DATA_PATH = original_data_path

    def test_random_forest_trains_and_evaluates(self) -> None:
        from train_model import randomForest

        data = {
            "Gender": ["Male", "Female", "Male", "Female"],
            "Married": ["Yes", "No", "Yes", "No"],
            "Dependents": ["0", "1", "0", "1"],
            "Education": ["Graduate", "Not Graduate", "Graduate", "Not Graduate"],
            "Self_Employed": ["No", "Yes", "No", "Yes"],
            "ApplicantIncome": [5000, 3000, 6000, 2500],
            "CoapplicantIncome": [0, 1500, 500, 1000],
            "LoanAmount": [100, 120, 130, 110],
            "Loan_Amount_Term": [360, 360, 360, 360],
            "Credit_History": [1.0, 0.0, 1.0, 0.0],
            "Property_Area": ["Urban", "Rural", "Semiurban", "Urban"],
            "Loan_Status": ["Y", "N", "Y", "N"],
        }
        df = pd.DataFrame(data)
        x = df[[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ]]
        y = df["Loan_Status"].map({"N": 0, "Y": 1})

        x_train = x.iloc[:2]
        x_test = x.iloc[2:]
        y_train = y.iloc[:2]
        y_test = y.iloc[2:]

        result = randomForest(x_train, x_test, y_train, y_test)

        self.assertEqual(result["model"], "Random Forest")
        self.assertIn("confusion_matrix", result)
        self.assertIn("classification_report", result)
        self.assertEqual(len(result["predictions"]), len(x_test))
        self.assertEqual(len(result["confusion_matrix"],), 2)
        self.assertIn("Approved", result["classification_report"])

    def test_xgboost_trains_and_evaluates(self) -> None:
        from train_model import xgboost

        data = {
            "Gender": ["Male", "Female", "Male", "Female"],
            "Married": ["Yes", "No", "Yes", "No"],
            "Dependents": ["0", "1", "0", "1"],
            "Education": ["Graduate", "Not Graduate", "Graduate", "Not Graduate"],
            "Self_Employed": ["No", "Yes", "No", "Yes"],
            "ApplicantIncome": [5000, 3000, 6000, 2500],
            "CoapplicantIncome": [0, 1500, 500, 1000],
            "LoanAmount": [100, 120, 130, 110],
            "Loan_Amount_Term": [360, 360, 360, 360],
            "Credit_History": [1.0, 0.0, 1.0, 0.0],
            "Property_Area": ["Urban", "Rural", "Semiurban", "Urban"],
            "Loan_Status": ["Y", "N", "Y", "N"],
        }
        df = pd.DataFrame(data)
        x = df[[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ]]
        y = df["Loan_Status"].map({"N": 0, "Y": 1})

        x_train = x.iloc[:2]
        x_test = x.iloc[2:]
        y_train = y.iloc[:2]
        y_test = y.iloc[2:]

        result = xgboost(x_train, x_test, y_train, y_test)

        self.assertEqual(result["model"], "XGBoost")
        self.assertIn("confusion_matrix", result)
        self.assertIn("classification_report", result)
        self.assertEqual(len(result["predictions"]), len(x_test))
        self.assertEqual(len(result["confusion_matrix"],), 2)
        self.assertIn("Approved", result["classification_report"])


if __name__ == "__main__":
    unittest.main()
