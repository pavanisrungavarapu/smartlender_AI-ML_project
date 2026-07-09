import unittest

from backend.app import create_app


class SmartLenderAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()

    def test_home_page_renders(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Smart Lender", response.data)
        self.assertIn(b"Loan Application", response.data)

    def test_prediction_endpoint_returns_result_page(self) -> None:
        response = self.client.post(
            "/predict",
            data={
                "gender": "Male",
                "married": "Yes",
                "dependents": "1",
                "education": "Graduate",
                "self_employed": "No",
                "applicant_income": "6500",
                "coapplicant_income": "2200",
                "loan_amount": "145",
                "loan_amount_term": "360",
                "credit_history": "1",
                "property_area": "Urban",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Result", response.data)
        self.assertIn(b"Submitted Profile", response.data)


if __name__ == "__main__":
    unittest.main()
