<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=MLOps%20Model%20Deployment%20Pipeline&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Automated%20retraining%20with%20drift%20detection%20and%20champion%2Fchallenger%20deployment&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/AUC-0.85+-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PSI_Drift-Detected-f59e0b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **An automated retraining pipeline that monitors data drift, versions models, and promotes challengers only when they beat the production champion.**

Production ML systems degrade over time as incoming data distributions shift away from the training set. This project wraps a churn prediction model in a full MLOps pipeline that validates data on ingestion, engineers features, trains and evaluates a GradientBoosting classifier, compares it against the current production model, and promotes it only if it exceeds a configurable performance threshold. Population Stability Index (PSI) monitors every numeric feature for drift, and a GitHub Actions workflow automates the entire cycle on every push to main.

```
Problem   →  Models degrade silently in production as data distributions shift
Solution  →  Automated pipeline with drift detection, versioning, and champion/challenger promotion
Impact    →  PSI-based retraining triggers, model registry with full audit trail, CI/CD via GitHub Actions
```

---

## Key results

| Metric | Value |
|--------|-------|
| Holdout ROC AUC | 0.85+ |
| PSI drift detected | 3 features above threshold |
| Model versions tracked | Timestamp-based registry |
| CI/CD | GitHub Actions on push to main |

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Data            │───▶│  Feature         │───▶│  Model           │
│  validation      │    │  engineering     │    │  training        │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                          ┌──────────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Holdout             │───▶│  Champion vs         │
              │  evaluation          │    │  challenger          │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                          ┌──────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Conditional         │───▶│  Drift               │
              │  promotion           │    │  monitoring (PSI)    │
              └──────────────────────┘    └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_19_mlops_pipeline/
├── .github/
│   └── workflows/
│       └── train.yml                  # CI/CD pipeline
├── data/
│   ├── generate_data.py               # Synthetic data generator
│   ├── train.csv                      # Training set
│   ├── test.csv                       # Holdout set
│   └── drift.csv                      # Shifted distribution set
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Loading, validation, schema checks
│   ├── model.py                       # Training, evaluation, registry, PSI
│   └── pipeline.py                    # End-to-end orchestration
├── notebooks/
│   ├── 01_eda.ipynb                   # Data validation and distributions
│   ├── 02_feature_engineering.ipynb   # Feature pipeline and PSI demo
│   ├── 03_modeling.ipynb              # Training and versioning
│   └── 04_evaluation.ipynb            # Monitoring and retraining triggers
├── artifacts/                         # Model registry (generated)
├── metrics/                           # Run logs (generated)
├── app.py                             # Streamlit dashboard
├── Dockerfile                         # Container definition
├── requirements.txt
├── index.html
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_19_mlops_pipeline

# Install dependencies
pip install -r requirements.txt

# Generate datasets
python data/generate_data.py

# Run the full pipeline
python -m src.pipeline

# Launch dashboard
streamlit run app.py
```

### Docker

```bash
docker build -t mlops-pipeline .
docker run -p 8501:8501 mlops-pipeline
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic telecom churn data |
| Train size | 5 000 rows |
| Test size | 1 500 rows |
| Drift set | 2 000 rows (shifted distributions) |
| Target | Binary churn (0/1) |
| Features | 19 (demographic, service, billing) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-006600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
</p>

---

## Methodology

<details>
<summary><b>Data validation</b></summary>

- Schema checks: column presence, dtype verification
- Null rate thresholds (max 5% per column)
- Range checks for numeric features (tenure >= 0, charges >= 0)
- Distribution drift detection against a reference dataset
</details>

<details>
<summary><b>Feature engineering</b></summary>

- Average monthly spend (TotalCharges / tenure)
- Tenure bucketing into lifecycle stages
- Internet service flag and service count aggregation
- StandardScaler for numeric, OneHotEncoder for categorical
</details>

<details>
<summary><b>Model training and versioning</b></summary>

- GradientBoosting classifier with 5-fold cross-validation
- Timestamp-based model versioning in a local registry
- Full metadata saved: parameters, metrics, timestamps
- Production pointer file for deployment tracking
</details>

<details>
<summary><b>PSI drift detection</b></summary>

- Population Stability Index computed per numeric feature
- Thresholds: < 0.10 (stable), 0.10-0.25 (investigate), > 0.25 (retrain)
- Drift dataset simulates distribution shift in tenure, charges, and contract type
</details>

<details>
<summary><b>Champion/challenger deployment</b></summary>

- New model compared against production on a primary metric (ROC AUC)
- Promotion only if challenger exceeds champion by a configurable threshold (default 0.005)
- Full audit trail of comparisons logged to JSON
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
