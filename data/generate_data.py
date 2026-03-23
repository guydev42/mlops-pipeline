"""Generate train, test, and drift datasets for the MLOps pipeline."""

import os
import sys

# Ensure project root is importable
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import generate_synthetic_churn_data


def main():
    data_dir = os.path.join(PROJECT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("Generating training data (5 000 rows)...")
    train_df = generate_synthetic_churn_data(n_samples=5000, seed=42, drift=False)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)

    print("Generating test data (1 500 rows)...")
    test_df = generate_synthetic_churn_data(n_samples=1500, seed=99, drift=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    print("Generating drift data (2 000 rows with shifted distributions)...")
    drift_df = generate_synthetic_churn_data(n_samples=2000, seed=77, drift=True)
    drift_df.to_csv(os.path.join(data_dir, "drift.csv"), index=False)

    print(f"Datasets saved to {data_dir}/")
    print(f"  train.csv  — {len(train_df)} rows, churn rate {train_df['Churn'].mean():.2%}")
    print(f"  test.csv   — {len(test_df)} rows, churn rate {test_df['Churn'].mean():.2%}")
    print(f"  drift.csv  — {len(drift_df)} rows, churn rate {drift_df['Churn'].mean():.2%}")


if __name__ == "__main__":
    main()
