from __future__ import annotations

from flask import Flask, render_template, request

from backend.services.prediction_service import PredictionService


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    service = PredictionService()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/predict")
    def predict() -> str:
        applicant = service.build_applicant(request.form)
        prediction = service.predict(applicant)
        return render_template(
            "result.html",
            result=prediction["result"],
            status=prediction["status"],
            confidence=prediction["confidence"],
            model_name=prediction["model_name"],
            applicant=applicant,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
