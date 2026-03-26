"""
Phase 5: Evaluation
====================
Input:  data/splits/X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
        models/lasso.pkl, random_forest.pkl, xgboost.pkl

Output: outputs/evaluation_report.txt   — metrics table (R², RMSE, MAE)
        outputs/residuals.png            — predicted vs actual for all 3 models
        outputs/shap_xgboost.png         — SHAP summary + bar (full test set)
        outputs/shap_rf.png              — SHAP summary + bar (500-row sample)
        outputs/lasso_coefficients.png   — signed coefficients ranked by magnitude
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
SPLITS_DIR  = os.path.join(BASE_DIR, "data", "splits_no_artist")
MODELS_DIR  = os.path.join(BASE_DIR, "models_no_artist")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs_no_artist")
RANDOM_STATE = 42
RF_SHAP_SAMPLE = 500

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_splits():
    out = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        with open(os.path.join(SPLITS_DIR, f"{name}.pkl"), "rb") as f:
            out[name] = pickle.load(f)
    return out["X_train"], out["X_test"], out["y_train"], out["y_test"]


def load_model(name):
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        for split_label, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
            preds = model.predict(X)
            rows.append({
                "Model":  name,
                "Split":  split_label,
                "R²":     model.score(X, y),
                "RMSE":   root_mean_squared_error(y, preds),
                "MAE":    mean_absolute_error(y, preds),
            })
    df = pd.DataFrame(rows).set_index(["Model", "Split"])
    return df


def save_report(metrics_df: pd.DataFrame) -> None:
    report = metrics_df.to_string(float_format=lambda x: f"{x:.4f}")

    print("\n" + "="*65)
    print("PHASE 5 EVALUATION REPORT")
    print("="*65)
    print(report)
    print("="*65 + "\n")

    path = os.path.join(OUTPUTS_DIR, "evaluation_report.txt")
    with open(path, "w") as f:
        f.write("PHASE 5 EVALUATION REPORT\n")
        f.write("="*65 + "\n")
        f.write(report + "\n")
    log.info(f"Evaluation report saved → {path}")


# ── Residual plots ────────────────────────────────────────────────────────────

def plot_residuals(models: dict, X_test, y_test) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        preds = model.predict(X_test)
        ax.scatter(y_test, preds, alpha=0.3, s=8, color="steelblue")
        lim = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lim, lim, "r--", linewidth=1.2, label="Perfect prediction")
        ax.set_xlabel("Actual streamScore")
        ax.set_ylabel("Predicted streamScore")
        ax.set_title(name)
        ax.legend(fontsize=8)

    fig.suptitle("Predicted vs Actual streamScore (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Residuals plot saved → {path}")


# ── Lasso coefficients ────────────────────────────────────────────────────────

def plot_lasso_coefficients(lasso, feature_names) -> None:
    coefs = pd.Series(lasso.coef_, index=feature_names)
    coefs = coefs[coefs != 0].sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, len(coefs) * 0.4)))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in coefs]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient value")
    ax.set_title("Lasso — Non-zero Coefficients\n(red = positive effect, blue = negative)")
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "lasso_coefficients.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Lasso coefficients plot saved → {path}")


# ── SHAP ──────────────────────────────────────────────────────────────────────

def plot_shap(model, X_test: pd.DataFrame, model_name: str,
              sample_n: int = None, filename: str = "shap.png") -> None:
    if sample_n and len(X_test) > sample_n:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_test), size=sample_n, replace=False)
        X_shap = X_test.iloc[idx]
        label = f"(n={sample_n} sample)"
    else:
        X_shap = X_test
        label = f"(n={len(X_test)})"

    log.info(f"Computing SHAP values for {model_name} {label}...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: beeswarm summary
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_shap, show=False, plot_size=None)
    axes[0].set_title(f"{model_name} — SHAP Beeswarm {label}")

    # Right: mean |SHAP| bar chart
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, plot_size=None)
    axes[1].set_title(f"{model_name} — Mean |SHAP| Feature Importance")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"SHAP plot saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test = load_splits()

    models = {
        "Lasso":         load_model("lasso"),
        "Random Forest": load_model("random_forest"),
        "XGBoost":       load_model("xgboost"),
    }

    log.info("Computing metrics...")
    metrics = compute_metrics(models, X_train, X_test, y_train, y_test)
    save_report(metrics)

    log.info("Plotting residuals...")
    plot_residuals(models, X_test, y_test)

    log.info("Plotting Lasso coefficients...")
    plot_lasso_coefficients(models["Lasso"], X_test.columns.tolist())

    log.info("Computing SHAP for XGBoost (full test set)...")
    plot_shap(
        models["XGBoost"], X_test,
        model_name="XGBoost",
        sample_n=None,
        filename="shap_xgboost.png"
    )

    log.info("Computing SHAP for Random Forest (500-row sample)...")
    plot_shap(
        models["Random Forest"], X_test,
        model_name="Random Forest",
        sample_n=RF_SHAP_SAMPLE,
        filename="shap_rf.png"
    )

    log.info("✅ Phase 5 complete.")


if __name__ == "__main__":
    main()
