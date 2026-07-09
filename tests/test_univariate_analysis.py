import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.univariate_analysis import plot_univariate_analysis, plot_multivariate_analysis


class UnivariateAnalysisTests(unittest.TestCase):
    def test_plot_univariate_analysis_creates_output_files(self) -> None:
        df = pd.DataFrame(
            {
                "ApplicantIncome": [1200, 2000, 3500, 4500, 5000],
                "CoapplicantIncome": [0, 1000, 800, 600, 1200],
                "Credit_History": [0, 1, 1, 0, 1],
                "Gender": ["Male", "Female", "Male", "Male", "Female"],
                "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate"],
                "Property_Area": ["Urban", "Rural", "Semiurban", "Urban", "Rural"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = plot_univariate_analysis(df, output_dir=Path(tmp_dir))

            self.assertTrue(outputs["continuous_plot"].exists())
            self.assertTrue(outputs["categorical_plot"].exists())
            self.assertEqual(
                outputs["summary"]["continuous_columns"],
                ["ApplicantIncome", "CoapplicantIncome", "Credit_History"],
            )

    def test_plot_multivariate_analysis_creates_output_file(self) -> None:
        df = pd.DataFrame(
            {
                "Property_Area": ["Urban", "Rural", "Semiurban", "Urban", "Rural"],
                "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate"],
                "Loan_Amount_Term": [360, 360, 180, 360, 180],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = plot_multivariate_analysis(df, output_dir=Path(tmp_dir))

            self.assertTrue(outputs["multivariate_plot"].exists())
            self.assertEqual(
                outputs["summary"]["required_columns"],
                ["Property_Area", "Education", "Loan_Amount_Term"],
            )


if __name__ == "__main__":
    unittest.main()
