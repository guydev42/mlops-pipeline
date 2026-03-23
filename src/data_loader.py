"""Data loading, validation, and synthetic data generation for the MLOps pipeline."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schema
# ---------------------------------------------------------------------------
EXPECTED_SCHEMA = {
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "float64",
    "Churn": "int64",
}

EXPECTED_COLUMNS = list(EXPECTED_SCHEMA.keys())

NULL_THRESHOLD = 0.05  # max 5 % nulls per column


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
class ValidationResult:
    """Container for data validation outcomes."""

    def __init__(self):
        self.checks: List[Dict] = []
        self.passed = True

    def add(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def to_dict(self) -> List[Dict]:
        return self.checks

    def summary(self) -> str:
        n_pass = sum(1 for c in self.checks if c["passed"])
        return f"{n_pass}/{len(self.checks)} checks passed"


def validate_schema(df: pd.DataFrame) -> ValidationResult:
    """Check that the dataframe matches the expected column set and dtypes."""
    result = ValidationResult()

    # Column presence
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    result.add(
        "columns_present",
        len(missing) == 0,
        f"missing={list(missing)}" if missing else "all columns present",
    )
    if extra:
        result.add("no_extra_columns", False, f"extra={list(extra)}")
    else:
        result.add("no_extra_columns", True, "no unexpected columns")

    # Dtype compatibility
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            ok = actual == expected_dtype
            result.add(f"dtype_{col}", ok, f"expected {expected_dtype}, got {actual}")

    return result


def validate_nulls(df: pd.DataFrame) -> ValidationResult:
    """Check that null rates stay below the threshold."""
    result = ValidationResult()
    for col in df.columns:
        rate = df[col].isna().mean()
        ok = rate <= NULL_THRESHOLD
        result.add(f"null_rate_{col}", ok, f"{rate:.4f} (threshold {NULL_THRESHOLD})")
    return result


def validate_distributions(
    df: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
) -> ValidationResult:
    """Basic range and distribution checks.

    If *reference* is provided, checks whether numeric column means have drifted
    more than 2 standard deviations from the reference mean.
    """
    result = ValidationResult()

    # Range checks for known columns
    if "tenure" in df.columns:
        ok = df["tenure"].min() >= 0
        result.add("tenure_nonneg", ok, f"min={df['tenure'].min()}")
    if "MonthlyCharges" in df.columns:
        ok = df["MonthlyCharges"].min() >= 0
        result.add("charges_nonneg", ok, f"min={df['MonthlyCharges'].min():.2f}")
    if "Churn" in df.columns:
        unique = set(df["Churn"].unique())
        ok = unique.issubset({0, 1})
        result.add("churn_binary", ok, f"unique={unique}")

    # Drift vs reference
    if reference is not None:
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if col in reference.columns:
                ref_mean = reference[col].mean()
                ref_std = reference[col].std()
                cur_mean = df[col].mean()
                if ref_std > 0:
                    z = abs(cur_mean - ref_mean) / ref_std
                    ok = z < 2.0
                    result.add(
                        f"drift_{col}",
                        ok,
                        f"z={z:.2f} (ref_mean={ref_mean:.2f}, cur_mean={cur_mean:.2f})",
                    )

    return result


def run_all_validations(
    df: pd.DataFrame, reference: Optional[pd.DataFrame] = None
) -> ValidationResult:
    """Run schema, null, and distribution validations."""
    combined = ValidationResult()
    for vr in [
        validate_schema(df),
        validate_nulls(df),
        validate_distributions(df, reference),
    ]:
        for c in vr.checks:
            combined.checks.append(c)
            if not c["passed"]:
                combined.passed = False
    return combined


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files from *data_dir*."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info("Loaded %d train rows, %d test rows", len(train_df), len(test_df))
    return train_df, test_df


def load_drift_data(data_dir: str) -> pd.DataFrame:
    """Load the drift dataset used for monitoring tests."""
    path = os.path.join(data_dir, "drift.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Drift data not found at {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic_churn_data(
    n_samples: int = 5000,
    churn_rate: float = 0.26,
    seed: int = 42,
    drift: bool = False,
) -> pd.DataFrame:
    """Generate synthetic telecom churn data.

    Parameters
    ----------
    n_samples : int
        Number of rows to generate.
    churn_rate : float
        Approximate target churn rate.
    seed : int
        Random seed for reproducibility.
    drift : bool
        If True, shift feature distributions to simulate data drift.
    """
    rng = np.random.RandomState(seed)

    gender = rng.choice(["Male", "Female"], n_samples)
    senior = rng.binomial(1, 0.16, n_samples)
    partner = rng.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])

    tenure = rng.exponential(scale=32, size=n_samples).astype(int).clip(0, 72)
    if drift:
        tenure = (tenure * 0.5).astype(int).clip(0, 72)  # shift toward lower tenure

    phone = rng.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    multi_lines = rng.choice(
        ["Yes", "No", "No phone service"], n_samples, p=[0.42, 0.48, 0.10]
    )

    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )
    if drift:
        internet = rng.choice(
            ["DSL", "Fiber optic", "No"], n_samples, p=[0.20, 0.60, 0.20]
        )

    no_internet = internet == "No"
    _yes_no_noint = lambda p: np.where(  # noqa: E731
        no_internet,
        "No internet service",
        rng.choice(["Yes", "No"], n_samples, p=[p, 1 - p]),
    )

    online_sec = _yes_no_noint(0.29)
    online_bak = _yes_no_noint(0.34)
    dev_protect = _yes_no_noint(0.34)
    tech_sup = _yes_no_noint(0.29)
    stream_tv = _yes_no_noint(0.38)
    stream_mov = _yes_no_noint(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.55, 0.21, 0.24],
    )
    if drift:
        contract = rng.choice(
            ["Month-to-month", "One year", "Two year"],
            n_samples,
            p=[0.72, 0.15, 0.13],
        )

    paperless = rng.choice(["Yes", "No"], n_samples, p=[0.59, 0.41])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    monthly = rng.normal(64.8, 30.0, n_samples).clip(18.0, 118.0)
    if drift:
        monthly = rng.normal(78.0, 25.0, n_samples).clip(18.0, 118.0)

    total = monthly * tenure + rng.normal(0, 50, n_samples)
    total = total.clip(0, None)

    # Churn label (correlated with tenure, contract, monthly charges)
    churn_prob = 0.15 + 0.25 * (tenure < 12) + 0.15 * (contract == "Month-to-month")
    churn_prob = churn_prob + 0.10 * (monthly > 80)
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    # Adjust overall rate
    churn_prob = churn_prob * (churn_rate / churn_prob.mean())
    churn_prob = np.clip(churn_prob, 0.01, 0.99)
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak,
            "DeviceProtection": dev_protect,
            "TechSupport": tech_sup,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": np.round(monthly, 2),
            "TotalCharges": np.round(total, 2),
            "Churn": churn,
        }
    )
    return df
