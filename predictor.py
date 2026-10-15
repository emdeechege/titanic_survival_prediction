"""Inference using the fitted notebook pipeline; no training is performed."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "models" / "titanic_model.joblib"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def load_model():
    return joblib.load(MODEL_PATH)


def prepare_features(data):
    missing = [name for name in FEATURES if name not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError("Provide at least one passenger.")

    features = data.loc[:, FEATURES].copy()
    for column in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        values = pd.to_numeric(features[column], errors="coerce")
        invalid = features[column].notna() & values.isna()
        if invalid.any() or np.isinf(values).any():
            raise ValueError(f"{column} must contain finite numbers or empty values.")
        if (values.dropna() < 0).any():
            raise ValueError(f"{column} cannot be negative.")
        if column in ["Pclass", "SibSp", "Parch"]:
            if (values.dropna() % 1 != 0).any():
                raise ValueError(f"{column} must contain whole numbers.")
        features[column] = values

    if not features["Pclass"].dropna().isin([1, 2, 3]).all():
        raise ValueError("Pclass must be 1, 2, or 3.")

    for column, allowed in [("Sex", ["male", "female"]), ("Embarked", ["S", "C", "Q"])]:
        values = features[column].astype("string").str.strip()
        values = values.str.lower() if column == "Sex" else values.str.upper()
        values = values.replace("", pd.NA)
        if not values.dropna().isin(allowed).all():
            raise ValueError(f"{column} must be one of: {', '.join(allowed)}.")
        # sklearn's fitted categorical imputer expects np.nan, not pd.NA.
        features[column] = values.astype(object).where(values.notna(), np.nan)

    return features


def predict_passengers(model, data):
    features = prepare_features(data)
    survival_index = list(model.classes_).index(1)
    result = pd.DataFrame(index=data.index)
    if "PassengerId" in data.columns:
        result["PassengerId"] = data["PassengerId"]
    result["Survived"] = model.predict(features).astype(int)
    result["SurvivalProbability"] = model.predict_proba(features)[:, survival_index]
    return result
