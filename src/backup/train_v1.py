"""
Phase 4: Model Training
=======================
Input:  data/splits/X_train.pkl, y_train.pkl
        data/splits/X_test.pkl,  y_test.pkl

Output: models/lasso.pkl
        models/random_forest.pkl
        models/xgboost.pkl
        outputs/training_results.txt

Models:
  1. Lasso         — linear baseline, alpha selected via 5-fold CV
  2. Random Forest — ensemble, hyperparams tuned via 5-fold CV
  3. XGBoost       — gradient boosting, hyperparams tuned via 5-fold CV

All models trained on scaled X_train (Lasso requires it;
tree models are unaffected but we use the single scaled copy for consistency).
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
SPLITS_DIR  = os.path.join(BASE_DIR, "data", "splits")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS     = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_splits():
    splits = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        path = os.path.join(SPLITS_DIR, f"{name}.pkl")
        with open(path, "rb") as f:
            splits[name] = pickle.load(f)
    log.info(f"Splits loaded — X_train: {splits['X_train'].shape}, X_test: {splits['X_test'].shape}")
    return splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"]


def evaluate(model, X_train, X_test, y_train, y_test, name):
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_rmse = root_mean_squared_error(y_train, train_pred)
    test_rmse  = root_mean_squared_error(y_test,  test_pred)
    train_r2   = model.score(X_train, y_train) if hasattr(model, "score") else float("nan")
    test_r2    = model.score(X_test,  y_test)  if hasattr(model, "score") else float("nan")

    log.info(f"{name} — Train RMSE: {train_rmse:.3f}, R²: {train_r2:.3f} | "
             f"Test RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")

    return {
        "model":      name,
        "train_rmse": train_rmse,
        "test_rmse":  test_rmse,
        "train_r2":   train_r2,
        "test_r2":    test_r2,
    }


def save_model(model, filename):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {path}")


# ── Model 1: Lasso ────────────────────────────────────────────────────────────

def train_lasso(X_train, y_train):
    """
    LassoCV selects the best alpha via 5-fold cross-validation.
    Alphas span a wide log-scale range to ensure the CV surface is fully explored.
    """
    log.info("Training Lasso (LassoCV, 5-fold)...")
    alphas = np.logspace(-4, 2, 100)
    model = LassoCV(
        alphas=alphas,
        cv=CV_FOLDS,
        max_iter=10_000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log.info(f"Lasso best alpha: {model.alpha_:.6f}")
    log.info(f"Lasso non-zero coefficients: {np.sum(model.coef_ != 0)} / {len(model.coef_)}")
    return model


# ── Model 2: Random Forest ────────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    """
    RandomForestRegressor with a modest grid search via cross-validation.
    n_estimators=500 is enough for a 23k-row dataset.
    min_samples_leaf=5 prevents overfitting on noisy streamScore targets.
    """
    log.info("Training Random Forest...")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    candidates = [
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5,  "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 20,   "min_samples_leaf": 5,  "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5,  "max_features": 0.5},
    ]

    best_score, best_params = -np.inf, None
    for params in candidates:
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        mean_r2 = scores.mean()
        log.info(f"  RF params {params} → CV R²: {mean_r2:.4f} ± {scores.std():.4f}")
        if mean_r2 > best_score:
            best_score, best_params = mean_r2, params

    log.info(f"Best RF params: {best_params} (CV R²: {best_score:.4f})")
    model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# ── Model 3: XGBoost ──────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    """
    XGBRegressor with a small grid search via cross-validation.
    early_stopping_rounds not used here — CV handles overfitting selection.
    subsample + colsample_bytree add stochastic regularisation.
    """
    log.info("Training XGBoost...")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    candidates = [
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,  "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.01,  "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.7, "colsample_bytree": 0.7},
    ]

    best_score, best_params = -np.inf, None
    for params in candidates:
        model = XGBRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        mean_r2 = scores.mean()
        log.info(f"  XGB params {params} → CV R²: {mean_r2:.4f} ± {scores.std():.4f}")
        if mean_r2 > best_score:
            best_score, best_params = mean_r2, params

    log.info(f"Best XGB params: {best_params} (CV R²: {best_score:.4f})")
    model = XGBRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ── Results table ─────────────────────────────────────────────────────────────

def save_results(results: list) -> None:
    df = pd.DataFrame(results).set_index("model")
    df = df[["train_r2", "test_r2", "train_rmse", "test_rmse"]]
    df.columns = ["Train R²", "Test R²", "Train RMSE", "Test RMSE"]

    print("\n" + "="*60)
    print("PHASE 4 TRAINING RESULTS")
    print("="*60)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("="*60 + "\n")

    out_path = os.path.join(OUTPUTS_DIR, "training_results.txt")
    with open(out_path, "w") as f:
        f.write("PHASE 4 TRAINING RESULTS\n")
        f.write("="*60 + "\n")
        f.write(df.to_string(float_format=lambda x: f"{x:.4f}"))
        f.write("\n")
    log.info(f"Results saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test = load_splits()

    results = []

    log.info("── Model 1: Lasso ──────────────────────────────────────")
    lasso = train_lasso(X_train, y_train)
    save_model(lasso, "lasso.pkl")
    results.append(evaluate(lasso, X_train, X_test, y_train, y_test, "Lasso"))

    log.info("── Model 2: Random Forest ──────────────────────────────")
    rf = train_random_forest(X_train, y_train)
    save_model(rf, "random_forest.pkl")
    results.append(evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest"))

    log.info("── Model 3: XGBoost ────────────────────────────────────")
    xgb = train_xgboost(X_train, y_train)
    save_model(xgb, "xgboost.pkl")
    results.append(evaluate(xgb, X_train, X_test, y_train, y_test, "XGBoost"))

    save_results(results)
    log.info("✅ Phase 4 complete.")


if __name__ == "__main__":
    main()
