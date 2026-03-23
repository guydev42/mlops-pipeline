"""Streamlit dashboard for the MLOps model deployment pipeline."""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import (
    load_training_data,
    load_drift_data,
    run_all_validations,
    generate_synthetic_churn_data,
)
from src.model import (
    ModelTrainer,
    ModelEvaluator,
    ModelRegistry,
    MetricsLogger,
    compute_psi,
    compare_models,
)
from src.pipeline import PipelineRun, PipelineConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MLOps pipeline dashboard", layout="wide")

DATA_DIR = os.path.join(PROJECT_DIR, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
METRICS_DIR = os.path.join(PROJECT_DIR, "metrics")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def cached_load_data():
    try:
        train_df, test_df = load_training_data(DATA_DIR)
        return train_df, test_df
    except FileNotFoundError:
        return None, None


@st.cache_data
def cached_load_drift():
    try:
        return load_drift_data(DATA_DIR)
    except FileNotFoundError:
        return None


def load_metrics_log():
    ml = MetricsLogger(os.path.join(METRICS_DIR, "metrics_log.json"))
    return ml.read_all()


def load_latest_run():
    path = os.path.join(METRICS_DIR, "latest_run.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_registry():
    reg = ModelRegistry(ARTIFACTS_DIR)
    return reg.list_versions()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Pipeline status",
        "Model registry",
        "Data drift monitor",
        "Model comparison",
        "Retrain",
    ],
)

# ---------------------------------------------------------------------------
# Page: Pipeline status
# ---------------------------------------------------------------------------
if page == "Pipeline status":
    st.title("Pipeline status")
    st.markdown("Current state of the last training pipeline run.")

    latest = load_latest_run()
    if latest is None:
        st.info("No pipeline runs found. Use the **Retrain** page to trigger a run.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", latest.get("status", "UNKNOWN"))
        col2.metric("Model version", latest.get("model_version", "—"))
        col3.metric("Finished at", latest.get("finished_at", "—")[:19])

        # Step-by-step breakdown
        st.subheader("Pipeline steps")
        steps = latest.get("steps", [])
        if steps:
            step_df = pd.DataFrame(steps)
            st.dataframe(step_df, use_container_width=True, hide_index=True)

            # Duration chart
            fig = px.bar(
                step_df,
                x="step",
                y="duration_s",
                color="status",
                color_discrete_map={"PASSED": "#22c55e", "WARNING": "#f59e0b", "FAILED": "#ef4444"},
                title="Step duration (seconds)",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Holdout metrics
        holdout = latest.get("holdout_metrics")
        if holdout:
            st.subheader("Holdout metrics")
            mcols = st.columns(len(holdout))
            for i, (k, v) in enumerate(holdout.items()):
                mcols[i].metric(k.upper(), f"{v:.4f}")

        # Validation summary
        validations = latest.get("validation", [])
        if validations:
            with st.expander("Data validation details"):
                val_df = pd.DataFrame(validations)
                st.dataframe(val_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Model registry
# ---------------------------------------------------------------------------
elif page == "Model registry":
    st.title("Model registry")
    st.markdown("All model versions with their metrics and timestamps.")

    versions = load_registry()
    if not versions:
        st.info("No models registered yet. Run the pipeline first.")
    else:
        # Production flag
        reg = ModelRegistry(ARTIFACTS_DIR)
        prod = reg.get_production_version()

        rows = []
        for v in versions:
            m = v.get("metrics", {})
            rows.append({
                "Version": v["version"],
                "Timestamp": v.get("timestamp", "")[:19],
                "AUC": m.get("roc_auc", 0),
                "F1": m.get("f1", 0),
                "Accuracy": m.get("accuracy", 0),
                "Precision": m.get("precision", 0),
                "Recall": m.get("recall", 0),
                "Production": "Yes" if v["version"] == prod else "",
            })
        reg_df = pd.DataFrame(rows)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        # Metrics over time
        if len(rows) > 1:
            st.subheader("Metrics over time")
            fig = go.Figure()
            for metric_name in ["AUC", "F1", "Accuracy"]:
                fig.add_trace(go.Scatter(
                    x=reg_df["Version"],
                    y=reg_df[metric_name],
                    mode="lines+markers",
                    name=metric_name,
                ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Score",
                xaxis_title="Model version",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Data drift monitor
# ---------------------------------------------------------------------------
elif page == "Data drift monitor":
    st.title("Data drift monitor")
    st.markdown("Population Stability Index (PSI) comparison between training and incoming data.")

    train_df, _ = cached_load_data()
    drift_df = cached_load_drift()

    if train_df is None or drift_df is None:
        st.warning("Training or drift data not found. Generate data first.")
    else:
        psi_scores = compute_psi(train_df, drift_df)

        st.subheader("PSI per feature")
        psi_df = pd.DataFrame([
            {"Feature": k, "PSI": v, "Status": "Significant shift" if v > 0.25 else ("Moderate shift" if v > 0.10 else "No shift")}
            for k, v in sorted(psi_scores.items(), key=lambda x: -x[1])
        ])
        st.dataframe(psi_df, use_container_width=True, hide_index=True)

        # Bar chart
        fig = px.bar(
            psi_df,
            x="Feature",
            y="PSI",
            color="Status",
            color_discrete_map={
                "No shift": "#22c55e",
                "Moderate shift": "#f59e0b",
                "Significant shift": "#ef4444",
            },
            title="PSI scores by feature",
        )
        fig.add_hline(y=0.25, line_dash="dash", line_color="#ef4444", annotation_text="Retrain threshold")
        fig.add_hline(y=0.10, line_dash="dot", line_color="#f59e0b", annotation_text="Investigation threshold")
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution overlays
        st.subheader("Feature distributions (train vs drift)")
        numeric_cols = sorted(psi_scores.keys())
        selected = st.selectbox("Select feature", numeric_cols)
        if selected:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=train_df[selected], name="Training", opacity=0.6,
                marker_color="#3B6FD4", nbinsx=40,
            ))
            fig2.add_trace(go.Histogram(
                x=drift_df[selected], name="Drift", opacity=0.6,
                marker_color="#E8C230", nbinsx=40,
            ))
            fig2.update_layout(
                barmode="overlay",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title=f"{selected} — PSI = {psi_scores.get(selected, 0):.4f}",
            )
            st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model comparison
# ---------------------------------------------------------------------------
elif page == "Model comparison":
    st.title("Model comparison")
    st.markdown("Champion vs challenger analysis.")

    versions = load_registry()
    if len(versions) < 2:
        st.info("Need at least two model versions for comparison. Run the pipeline multiple times.")
        if len(versions) == 1:
            st.write("Current model metrics:")
            st.json(versions[0].get("metrics", {}))
    else:
        version_names = [v["version"] for v in versions]
        col1, col2 = st.columns(2)
        champion_v = col1.selectbox("Champion (A)", version_names, index=len(version_names) - 2)
        challenger_v = col2.selectbox("Challenger (B)", version_names, index=len(version_names) - 1)

        champ_meta = next(v for v in versions if v["version"] == champion_v)
        chall_meta = next(v for v in versions if v["version"] == challenger_v)

        comparison = compare_models(
            champ_meta.get("metrics", {}),
            chall_meta.get("metrics", {}),
        )

        # Result banner
        if comparison["promote_challenger"]:
            st.success(
                f"Challenger wins by {comparison['difference']:+.4f} on {comparison['primary_metric']}. "
                "Promotion recommended."
            )
        else:
            st.warning(
                f"Champion holds. Difference: {comparison['difference']:+.4f} "
                f"(threshold: {comparison['threshold']})."
            )

        # Side-by-side metrics
        st.subheader("Metrics comparison")
        metric_names = list(champ_meta.get("metrics", {}).keys())
        comp_rows = []
        for m in metric_names:
            va = champ_meta["metrics"].get(m, 0)
            vb = chall_meta["metrics"].get(m, 0)
            comp_rows.append({
                "Metric": m,
                "Champion": f"{va:.4f}",
                "Challenger": f"{vb:.4f}",
                "Diff": f"{vb - va:+.4f}",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # Radar chart
        if metric_names:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[champ_meta["metrics"].get(m, 0) for m in metric_names],
                theta=metric_names,
                fill="toself",
                name=f"Champion ({champion_v})",
                line_color="#3B6FD4",
            ))
            fig.add_trace(go.Scatterpolar(
                r=[chall_meta["metrics"].get(m, 0) for m in metric_names],
                theta=metric_names,
                fill="toself",
                name=f"Challenger ({challenger_v})",
                line_color="#E8C230",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                title="Metric comparison (radar)",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Retrain
# ---------------------------------------------------------------------------
elif page == "Retrain":
    st.title("Retrain pipeline")
    st.markdown("Trigger a new pipeline run or regenerate data.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Generate data")
        if st.button("Regenerate datasets"):
            with st.spinner("Generating train, test, and drift datasets..."):
                train_df = generate_synthetic_churn_data(5000, seed=int(time.time()) % 10000)
                test_df = generate_synthetic_churn_data(1500, seed=int(time.time()) % 10000 + 1)
                drift_df = generate_synthetic_churn_data(2000, seed=int(time.time()) % 10000 + 2, drift=True)

                os.makedirs(DATA_DIR, exist_ok=True)
                train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
                test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
                drift_df.to_csv(os.path.join(DATA_DIR, "drift.csv"), index=False)
            st.success(f"Generated {len(train_df)} train, {len(test_df)} test, {len(drift_df)} drift rows.")
            st.cache_data.clear()

    with col2:
        st.subheader("Run pipeline")
        if st.button("Trigger retraining"):
            with st.spinner("Running full pipeline..."):
                try:
                    config = PipelineConfig(
                        data_dir="data",
                        artifacts_dir="artifacts",
                        metrics_dir="metrics",
                    )
                    run = PipelineRun(config)
                    result = run.run()
                    st.success(f"Pipeline finished — status: {result['status']}, version: {result.get('model_version', '—')}")
                    st.json(result.get("holdout_metrics", {}))
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

    # Run history
    st.subheader("Run history")
    entries = load_metrics_log()
    if entries:
        history_rows = []
        for e in entries:
            hm = e.get("holdout_metrics", {})
            history_rows.append({
                "Logged at": e.get("logged_at", "")[:19],
                "Status": e.get("status", ""),
                "Version": e.get("model_version", ""),
                "AUC": hm.get("roc_auc", ""),
                "F1": hm.get("f1", ""),
                "Drift detected": e.get("drift_detected", ""),
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No runs recorded yet.")
