"""Model training, evaluation, versioning, and drift detection."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET = "Churn"


class FeatureEngineer:
    """Create derived features and prepare the modelling matrix."""

    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["AvgMonthlySpend"] = np.where(
            out["tenure"] > 0,
            out["TotalCharges"] / out["tenure"],
            out["MonthlyCharges"],
        )
        out["tenure_bucket"] = pd.cut(
            out["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12", "12-24", "24-48", "48-72"],
            include_lowest=True,
        ).astype(str)
        out["has_internet"] = (out["InternetService"] != "No").astype(int)
        out["num_services"] = 0
        for col in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            out["num_services"] += (out[col] == "Yes").astype(int)
        return out

    @staticmethod
    def build_preprocessor() -> ColumnTransformer:
        numeric_cols = NUMERIC_FEATURES + ["AvgMonthlySpend", "has_internet", "num_services"]
        categorical_cols = CATEGORICAL_FEATURES + ["tenure_bucket"]
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ],
            remainder="drop",
        )


# ---------------------------------------------------------------------------
# Drift detection — Population Stability Index (PSI)
# ---------------------------------------------------------------------------
def _psi_bucket(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two 1-d arrays."""
    eps = 1e-4
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Optional[List[str]] = None,
    bins: int = 10,
) -> Dict[str, float]:
    """Return PSI per numeric column.

    PSI thresholds (common rule of thumb):
        < 0.10  — no significant shift
        0.10–0.25  — moderate shift, investigate
        > 0.25  — significant shift, retrain
    """
    if columns is None:
        columns = list(reference.select_dtypes(include="number").columns)

    results = {}
    for col in columns:
        if col in reference.columns and col in current.columns:
            ref_vals = reference[col].dropna().values.astype(float)
            cur_vals = current[col].dropna().values.astype(float)
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                results[col] = _psi_bucket(ref_vals, cur_vals, bins=bins)
    return results


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------
class ModelTrainer:
    """Wraps the full sklearn pipeline: preprocessing + classifier."""

    def __init__(self, model_params: Optional[Dict] = None):
        self.model_params = model_params or {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        }
        self.pipeline: Optional[Pipeline] = None
        self.feature_engineer = FeatureEngineer()

    def fit(self, df: pd.DataFrame) -> "ModelTrainer":
        df_feat = self.feature_engineer.add_derived_features(df)
        X = df_feat.drop(columns=[TARGET])
        y = df_feat[TARGET]

        preprocessor = self.feature_engineer.build_preprocessor()
        clf = GradientBoostingClassifier(**self.model_params)
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        self.pipeline.fit(X, y)
        logger.info("Model trained on %d samples", len(X))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = self.feature_engineer.add_derived_features(df)
        X = df_feat.drop(columns=[TARGET], errors="ignore")
        return self.pipeline.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = self.feature_engineer.add_derived_features(df)
        X = df_feat.drop(columns=[TARGET], errors="ignore")
        return self.pipeline.predict_proba(X)[:, 1]

    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> Dict[str, float]:
        df_feat = self.feature_engineer.add_derived_features(df)
        X = df_feat.drop(columns=[TARGET])
        y = df_feat[TARGET]

        preprocessor = self.feature_engineer.build_preprocessor()
        clf = GradientBoostingClassifier(**self.model_params)
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        return {"mean_auc": float(scores.mean()), "std_auc": float(scores.std())}


# ---------------------------------------------------------------------------
# Model evaluator
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """Evaluate a trained model on a holdout set."""

    @staticmethod
    def evaluate(trainer: ModelTrainer, df: pd.DataFrame) -> Dict[str, float]:
        y_true = df[TARGET].values
        y_pred = trainer.predict(df)
        y_prob = trainer.predict_proba(df)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }
        logger.info("Evaluation: %s", metrics)
        return metrics


# ---------------------------------------------------------------------------
# Model registry (local simulation)
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Simple file-system model registry with timestamp versioning."""

    def __init__(self, registry_dir: str = "artifacts"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)

    def _version_tag(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def save_model(
        self,
        trainer: ModelTrainer,
        metrics: Dict[str, float],
        tag: Optional[str] = None,
    ) -> str:
        version = tag or self._version_tag()
        model_dir = os.path.join(self.registry_dir, version)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(trainer.pipeline, model_path)

        meta = {
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "model_params": trainer.model_params,
        }
        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved model version %s to %s", version, model_dir)
        return version

    def load_model(self, version: str) -> Tuple[Pipeline, Dict]:
        model_dir = os.path.join(self.registry_dir, version)
        pipeline = joblib.load(os.path.join(model_dir, "model.joblib"))
        with open(os.path.join(model_dir, "metadata.json")) as f:
            meta = json.load(f)
        return pipeline, meta

    def list_versions(self) -> List[Dict]:
        versions = []
        if not os.path.exists(self.registry_dir):
            return versions
        for name in sorted(os.listdir(self.registry_dir)):
            meta_path = os.path.join(self.registry_dir, name, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    versions.append(json.load(f))
        return versions

    def get_production_version(self) -> Optional[str]:
        prod_path = os.path.join(self.registry_dir, "production.json")
        if os.path.exists(prod_path):
            with open(prod_path) as f:
                return json.load(f).get("version")
        return None

    def promote_to_production(self, version: str) -> None:
        prod_path = os.path.join(self.registry_dir, "production.json")
        with open(prod_path, "w") as f:
            json.dump({"version": version, "promoted_at": datetime.utcnow().isoformat()}, f, indent=2)
        logger.info("Promoted version %s to production", version)


# ---------------------------------------------------------------------------
# A/B model comparison
# ---------------------------------------------------------------------------
def compare_models(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    primary_metric: str = "roc_auc",
    threshold: float = 0.005,
) -> Dict[str, Any]:
    """Compare champion (A) vs challenger (B).

    Returns a dict with the comparison outcome.  The challenger is promoted
    only if it beats the champion by at least *threshold* on the primary metric.
    """
    val_a = metrics_a.get(primary_metric, 0)
    val_b = metrics_b.get(primary_metric, 0)
    diff = val_b - val_a

    promote = diff > threshold
    return {
        "champion_metric": val_a,
        "challenger_metric": val_b,
        "difference": diff,
        "threshold": threshold,
        "primary_metric": primary_metric,
        "promote_challenger": promote,
        "all_champion": metrics_a,
        "all_challenger": metrics_b,
    }


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------
class MetricsLogger:
    """Append pipeline run metrics to a JSON-lines file."""

    def __init__(self, log_path: str = "metrics/metrics_log.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, entry: Dict) -> None:
        entry["logged_at"] = datetime.utcnow().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self) -> List[Dict]:
        entries = []
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries
