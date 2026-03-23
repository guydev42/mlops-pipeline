"""End-to-end MLOps pipeline orchestration."""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

# Ensure project root is importable when running as a module
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_training_data, load_drift_data, run_all_validations
from src.model import (
    ModelTrainer,
    ModelEvaluator,
    ModelRegistry,
    MetricsLogger,
    FeatureEngineer,
    compare_models,
    compute_psi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """Central configuration for a pipeline run."""

    def __init__(
        self,
        data_dir: str = "data",
        artifacts_dir: str = "artifacts",
        metrics_dir: str = "metrics",
        primary_metric: str = "roc_auc",
        promotion_threshold: float = 0.005,
        psi_retrain_threshold: float = 0.25,
    ):
        self.data_dir = os.path.join(PROJECT_DIR, data_dir)
        self.artifacts_dir = os.path.join(PROJECT_DIR, artifacts_dir)
        self.metrics_dir = os.path.join(PROJECT_DIR, metrics_dir)
        self.primary_metric = primary_metric
        self.promotion_threshold = promotion_threshold
        self.psi_retrain_threshold = psi_retrain_threshold


class PipelineRun:
    """Orchestrates one full training/evaluation/promotion cycle."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.registry = ModelRegistry(self.config.artifacts_dir)
        self.metrics_logger = MetricsLogger(
            os.path.join(self.config.metrics_dir, "metrics_log.json")
        )
        self.run_log: Dict = {"steps": [], "started_at": datetime.utcnow().isoformat()}

    # -----------------------------------------------------------------------
    def _log_step(self, name: str, status: str, detail: str = "", duration: float = 0):
        entry = {
            "step": name,
            "status": status,
            "detail": detail,
            "duration_s": round(duration, 3),
        }
        self.run_log["steps"].append(entry)
        logger.info("Step [%s] %s — %s (%.2fs)", name, status, detail, duration)

    # -----------------------------------------------------------------------
    # Step 1: Validate data
    # -----------------------------------------------------------------------
    def step_validate_data(self) -> pd.DataFrame:
        t0 = time.time()
        train_df, test_df = load_training_data(self.config.data_dir)
        validation = run_all_validations(train_df)
        duration = time.time() - t0

        if validation.passed:
            self._log_step("validate_data", "PASSED", validation.summary(), duration)
        else:
            self._log_step("validate_data", "WARNING", validation.summary(), duration)

        self.run_log["validation"] = validation.to_dict()
        self._train_df = train_df
        self._test_df = test_df
        return train_df

    # -----------------------------------------------------------------------
    # Step 2: Engineer features
    # -----------------------------------------------------------------------
    def step_engineer_features(self) -> pd.DataFrame:
        t0 = time.time()
        fe = FeatureEngineer()
        self._train_feat = fe.add_derived_features(self._train_df)
        self._test_feat = fe.add_derived_features(self._test_df)
        n_new = len(self._train_feat.columns) - len(self._train_df.columns)
        duration = time.time() - t0
        self._log_step(
            "engineer_features",
            "PASSED",
            f"{n_new} derived features added",
            duration,
        )
        return self._train_feat

    # -----------------------------------------------------------------------
    # Step 3: Train model
    # -----------------------------------------------------------------------
    def step_train_model(self) -> ModelTrainer:
        t0 = time.time()
        self._trainer = ModelTrainer()
        self._trainer.fit(self._train_df)
        cv = self._trainer.cross_validate(self._train_df, cv=5)
        duration = time.time() - t0
        self._log_step(
            "train_model",
            "PASSED",
            f"CV AUC={cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}",
            duration,
        )
        self.run_log["cv_metrics"] = cv
        return self._trainer

    # -----------------------------------------------------------------------
    # Step 4: Evaluate on holdout
    # -----------------------------------------------------------------------
    def step_evaluate(self) -> Dict:
        t0 = time.time()
        self._metrics = ModelEvaluator.evaluate(self._trainer, self._test_df)
        duration = time.time() - t0
        self._log_step(
            "evaluate",
            "PASSED",
            f"AUC={self._metrics['roc_auc']:.4f}, F1={self._metrics['f1']:.4f}",
            duration,
        )
        self.run_log["holdout_metrics"] = self._metrics
        return self._metrics

    # -----------------------------------------------------------------------
    # Step 5: Compare with production model
    # -----------------------------------------------------------------------
    def step_compare_with_production(self) -> Dict:
        t0 = time.time()
        prod_version = self.registry.get_production_version()

        if prod_version is None:
            self._comparison = {
                "promote_challenger": True,
                "reason": "no production model exists",
            }
            duration = time.time() - t0
            self._log_step(
                "compare_models",
                "PASSED",
                "No production model — challenger auto-promoted",
                duration,
            )
            return self._comparison

        # Load production metadata
        _, prod_meta = self.registry.load_model(prod_version)
        prod_metrics = prod_meta.get("metrics", {})

        self._comparison = compare_models(
            prod_metrics,
            self._metrics,
            primary_metric=self.config.primary_metric,
            threshold=self.config.promotion_threshold,
        )
        duration = time.time() - t0
        outcome = "PROMOTE" if self._comparison["promote_challenger"] else "KEEP"
        self._log_step(
            "compare_models",
            "PASSED",
            f"{outcome} — diff={self._comparison['difference']:+.4f}",
            duration,
        )
        return self._comparison

    # -----------------------------------------------------------------------
    # Step 6: Promote if better
    # -----------------------------------------------------------------------
    def step_promote(self) -> str:
        t0 = time.time()
        version = self.registry.save_model(self._trainer, self._metrics)

        if self._comparison.get("promote_challenger", False):
            self.registry.promote_to_production(version)
            status_detail = f"version {version} promoted to production"
        else:
            status_detail = f"version {version} saved but NOT promoted"

        duration = time.time() - t0
        self._log_step("promote", "PASSED", status_detail, duration)
        self.run_log["model_version"] = version
        return version

    # -----------------------------------------------------------------------
    # Full run
    # -----------------------------------------------------------------------
    def run(self) -> Dict:
        """Execute the full pipeline end-to-end."""
        logger.info("=" * 60)
        logger.info("PIPELINE RUN STARTED")
        logger.info("=" * 60)

        self.step_validate_data()
        self.step_engineer_features()
        self.step_train_model()
        self.step_evaluate()
        self.step_compare_with_production()
        version = self.step_promote()

        # Check drift if drift data exists
        try:
            drift_df = load_drift_data(self.config.data_dir)
            psi_scores = compute_psi(self._train_df, drift_df)
            self.run_log["psi_scores"] = psi_scores
            max_psi = max(psi_scores.values()) if psi_scores else 0
            needs_retrain = max_psi > self.config.psi_retrain_threshold
            self.run_log["drift_detected"] = needs_retrain
            self.run_log["max_psi"] = max_psi
            logger.info(
                "Drift check: max PSI=%.4f, retrain=%s", max_psi, needs_retrain
            )
        except FileNotFoundError:
            logger.info("No drift dataset found — skipping drift check")

        self.run_log["finished_at"] = datetime.utcnow().isoformat()
        self.run_log["status"] = "SUCCESS"

        # Persist run log
        self.metrics_logger.log(self.run_log)

        # Save latest run summary
        summary_path = os.path.join(self.config.metrics_dir, "latest_run.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(self.run_log, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("PIPELINE RUN COMPLETE — version %s", version)
        logger.info("=" * 60)
        return self.run_log


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = PipelineRun().run()
    print(json.dumps({"status": result["status"], "version": result.get("model_version")}, indent=2))
