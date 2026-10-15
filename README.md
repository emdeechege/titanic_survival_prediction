# Titanic deployment

## [Open the live app →](https://titanicpredict-eipq.onrender.com/)

**Deployed on Render:** [titanicpredict-eipq.onrender.com](https://titanicpredict-eipq.onrender.com/)

A standalone FastAPI app with a browser page. The existing fitted model is
loaded once at startup; requests run inference without retraining.

## Run

Use Python 3.11. Open a terminal in the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows, activate with `.venv\Scripts\activate` instead.

- Web page: http://localhost:8000
- Interactive API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Publish on Render

Push this project to GitHub, including `models/titanic_model.joblib`,
`static/index.html`, and the hidden `.python-version` file.
The `.python-version` file selects Python 3.11 for the model's pinned dependencies.

In Render, select **New → Web Service**, connect the repository, and use:

| Setting | Value |
| --- | --- |
| Language / runtime | Python 3 |
| Root Directory | Leave empty for this standalone repository |
| Build Command | `python -m pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

If you instead place this project inside a larger repository, set Root Directory
to the subfolder containing `app.py`. Keep `.python-version` in the service root.
If you previously set a `PYTHON_VERSION` environment variable in Render, remove
it so `.python-version` controls the version.

Choose your instance plan and deploy. When Render reports the service is live,
open its `https://<your-service>.onrender.com` URL to use the passenger form;
append `/docs` to try the prediction API or `/health` to check readiness.

Render provides the `PORT` variable; the start command uses it directly:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

`python app.py` also reads `PORT`, defaulting to 8000 locally.
The deployed app is available at [titanicpredict-eipq.onrender.com](https://titanicpredict-eipq.onrender.com/).

## Trigger a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Pclass":1,"Sex":"female","Age":12,"SibSp":2,"Parch":1,"Fare":54.72,"Embarked":"S"}'
```

Response fields are `survived` (0 or 1) and `survival_probability` (0 to 1).
`POST /predict/batch` accepts an array of 1–10,000 passenger objects and returns
an array of predictions in the same order. Extra fields such as PassengerId
must be removed before submitting API requests.

Pclass (1/2/3) and Sex (male/female) are required. SibSp and Parch default to 0.
Age, Fare and Embarked can be omitted or set to null to use the fitted imputers.
Embarked accepts S, C or Q. Numeric values must be nonnegative and finite;
SibSp and Parch must be whole numbers. Invalid requests, including numbers too
large for the fitted model to process, receive HTTP 422.

## Included files

```text
titanic_prediction_model/
  app.py                     FastAPI routes and request validation
  predictor.py               Saved pipeline inference
  static/index.html          Responsive passenger form
  models/titanic_model.joblib Original fitted pipeline, copied unchanged
  data/                      Original train, test and gender submission CSVs
  notebooks/titanic.ipynb     Archived training and analysis notebook
  requirements.txt           Runtime dependencies
  requirements-dev.txt       Runtime dependencies plus test client
  .python-version            Python 3.11 selection for Render
  tests/test_deployment.py   API and prediction checks
```

The included notebook is an archival copy and retains its original Windows
paths and `../Models` capitalization. To rerun it here, first change its dataset
path to `../data/train.csv` and model paths to `../models/titanic_model.joblib`.
The notebook also requires Jupyter, matplotlib, seaborn, xgboost, and optuna;
these are not included in the deployment or test requirements. It records a
different original Python environment and is retained for reference, rather
than as a reproducible training workflow. The data and notebook are not served
by the API.

The saved artifact identifies scikit-learn 1.3.2 and contains a random forest
with 954 trees, maximum depth 29, minimum split size 15 and minimum leaf size 1.
Its preprocessing pipeline handles imputation, scaling and one-hot encoding.
The dependency pins match the local environment that loads this artifact;
keep them when deploying it. Python 3.12+ does not support these dependency
pins. Only replace the model with trusted artifacts, as joblib loading executes code.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m unittest discover -s tests -v
```

The checks cover the page, health endpoint, request validation, numeric overflow,
batch limits, missing values and defaults, and agreement with the fitted pipeline
for all 418 original test passengers.

## Page context and statistics

The page includes voyage facts, a summary of the labeled passenger dataset,
survival rates by class, input explanations, and guidance on interpreting an
estimate. It contains no navigation links to API documentation or external sites.

Dataset figures are a static snapshot calculated from `data/train.csv`: 891
records, 342 survivors (38.4%), and 177 missing ages. Survival by class is
136/216 (63.0%), 87/184 (47.3%), and 119/491 (24.2%). Recalculate these figures
in `static/index.html` if the dataset changes. These summarize the full labeled
dataset, rather than only the subset used to fit the model, and do not represent
everyone aboard the ship.

Historical figures are credited in plain text on the page to
[Royal Museums Greenwich, RMS Titanic facts](https://www.rmg.co.uk/stories/maritime-history/rms-titanic-facts).
The API endpoints remain available for integrations, including `/docs` for developers.

References: [FastAPI deployment](https://fastapi.tiangolo.com/deployment/manually/)
and [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).
Render references: [FastAPI deployment](https://render.com/docs/deploy-fastapi),
[Python version](https://render.com/docs/python-version), and
[Blueprint configuration](https://render.com/docs/blueprint-spec).
