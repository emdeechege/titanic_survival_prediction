import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app
from predictor import FEATURES


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def passenger(self):
        return {"Pclass": 1, "Sex": "female", "Age": 12, "SibSp": 2,
                "Parch": 1, "Fare": 54.72, "Embarked": "S"}

    def test_webpage_and_health(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="passenger-form"', response.text)
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_page_ignores_download_range_headers(self):
        response = self.client.get("/", headers={"Range": "bytes=0-10,20-30"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertEqual(response.text, (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text())

    def test_prediction_matches_saved_pipeline(self):
        data = self.passenger()
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        model = app.state.model
        frame = pd.DataFrame([data])[FEATURES]
        self.assertEqual(response.json()["survived"], int(model.predict(frame)[0]))
        self.assertAlmostEqual(response.json()["survival_probability"], model.predict_proba(frame)[0, 1])

    def test_unknown_values_use_saved_imputers(self):
        data = self.passenger()
        data.update(Age=None, Fare=None, Embarked=None)
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(0 <= response.json()["survival_probability"] <= 1)

    def test_omitted_values_match_explicit_defaults(self):
        minimal = {"Pclass": 3, "Sex": "male"}
        explicit = {**minimal, "Age": None, "Fare": None, "Embarked": None, "SibSp": 0, "Parch": 0}
        response = self.client.post("/predict", json=minimal)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.client.post("/predict", json=explicit).json())

    def test_nonfinite_json_numbers_are_rejected(self):
        for value in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(value=value):
                response = self.client.post(
                    "/predict", content='{"Pclass":1,"Sex":"female","Age":' + value + '}',
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"][0]["type"], "finite_number")

    def test_unrepresentable_model_numbers_are_rejected(self):
        for field in ["Age", "Fare", "SibSp", "Parch"]:
            with self.subTest(field=field):
                response = self.client.post("/predict", json={**self.passenger(), field: 10 ** 100})
                self.assertEqual(response.status_code, 422)

    def test_batch_limit_is_enforced(self):
        response = self.client.post("/predict/batch", json=[self.passenger()] * 10001)
        self.assertEqual(response.status_code, 422)

    def test_invalid_requests_are_rejected(self):
        for change in [{"Pclass": 4}, {"Age": -1}, {"Fare": "invalid"},
                       {"SibSp": 1.5}, {"Parch": -1}, {"Sex": "unknown"},
                       {"Embarked": "unknown"}, {"Age": "NaN"}, {"Fare": "Infinity"}, {"extra": 1}]:
            with self.subTest(change=change):
                self.assertEqual(self.client.post("/predict", json={**self.passenger(), **change}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={}).status_code, 422)
        self.assertEqual(self.client.post("/predict/batch", json=[]).status_code, 422)

    def test_all_original_test_passengers_match_pipeline(self):
        frame = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "test.csv")[FEATURES]
        records = frame.astype(object).where(frame.notna(), None).to_dict("records")
        response = self.client.post("/predict/batch", json=records)
        self.assertEqual(response.status_code, 200)
        predictions = response.json()
        self.assertEqual(len(predictions), 418)
        np.testing.assert_array_equal([r["survived"] for r in predictions], app.state.model.predict(frame))
        np.testing.assert_allclose([r["survival_probability"] for r in predictions], app.state.model.predict_proba(frame)[:, 1])


if __name__ == "__main__":
    unittest.main()
