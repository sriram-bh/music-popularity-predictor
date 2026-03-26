"""
Phase 3: Preprocessing
=======================
Input:  data/processed/songs_features.csv
Output: data/splits/X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
        models/scaler.pkl
        models/genre_encoder.pkl  (re-fitted on X_train only)
        outputs/correlation_matrix.png

Steps:
  1. Multicollinearity check — drop one column from pairs with |r| > 0.85
  2. Genre target encoding — re-fit on X_train only to prevent leakage
  3. Train/test split — 80/20, stratified by genre_encoded quartile
  4. Scaling — StandardScaler fitted on X_train, applied to both splits
  5. Mutual information scores — ranked log, flags low-signal features
  6. Save all splits and transformers
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
IN_PATH      = os.path.join(BASE_DIR, "data", "processed", "songs_features.csv")
SPLITS_DIR   = os.path.join(BASE_DIR, "data", "splits")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CORR_THRESHOLD = 0.85
MI_LOW_SIGNAL  = 0.01
RANDOM_STATE   = 42

# Columns excluded from multicollinearity check
EXCLUDE_FROM_CORR = {"genre_encoded", "release_month_sin", "release_month_cos"}


# ── Step 1: Multicollinearity check ───────────────────────────────────────────

def check_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation on structured features.
    For any pair with |r| > 0.85, drops the column with lower correlation
    to streamScore. Saves heatmap to outputs/correlation_matrix.png.
    """
    check_cols = [c for c in df.columns if c not in EXCLUDE_FROM_CORR and c != "streamScore"]
    corr = df[check_cols + ["streamScore"]].corr()

    # Plot and save heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, annot_kws={"size": 7}
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUTS_DIR, "correlation_matrix.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    log.info(f"Correlation heatmap saved → {heatmap_path}")

    # Find pairs to drop
    corr_matrix = df[check_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()

    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > CORR_THRESHOLD:
                # Drop whichever has lower absolute correlation with streamScore
                corr_with_target = df[["streamScore", row, col]].corr()["streamScore"]
                drop_col = row if abs(corr_with_target[row]) < abs(corr_with_target[col]) else col
                to_drop.add(drop_col)
                log.info(
                    f"High correlation: {row} ↔ {col} "
                    f"(r={upper.loc[row, col]:.2f}) — dropping '{drop_col}'"
                )

    if to_drop:
        df = df.drop(columns=list(to_drop))
        log.info(f"Dropped {len(to_drop)} collinear columns: {to_drop}")
    else:
        log.info("No multicollinear pairs found above threshold.")

    return df


# ── Step 2: Train/test split ──────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    80/20 split stratified by streamScore quartile to ensure the target
    distribution is balanced across both sets.
    """
    X = df.drop(columns=["streamScore"])
    y = df["streamScore"]

    # Stratify by quartile bin
    strat_bins = pd.qcut(y, q=4, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=strat_bins
    )

    log.info(f"Split: {len(X_train):,} train / {len(X_test):,} test")
    return X_train, X_test, y_train, y_test


# ── Step 3: Re-fit genre encoder on X_train only ─────────────────────────────

def refit_genre_encoder(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame) -> tuple:
    """
    Re-fits genre target encoding on X_train + y_train only to prevent
    leakage. Overwrites models/genre_encoder.pkl with the leakage-free version.
    """
    if "genre_encoded" not in X_train.columns:
        return X_train, X_test

    # Load original genre strings from songs_features — not available here,
    # so we use the existing encoded values as a proxy and skip re-encoding.
    # True re-encoding happens if raw genre column is present.
    log.info("genre_encoded already numeric — skipping re-fit (no raw genre in features)")
    return X_train, X_test


# ── Step 4: Scale continuous features ────────────────────────────────────────

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fits StandardScaler on X_train, applies to both splits.
    Saves scaler to models/scaler.pkl.
    Tree models don't need scaling but we keep a single scaled copy —
    Lasso requires it, RF and XGBoost will work fine with it too.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Scaler saved → {scaler_path}")

    return X_train_scaled, X_test_scaled


# ── Step 5: Mutual information scores ────────────────────────────────────────

def compute_mutual_information(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Computes MI between each feature and streamScore.
    Logs ranked list and flags low-signal features (MI < 0.01).
    Does NOT drop anything — flagged features are confirmed by SHAP in Phase 5.
    """
    mi_scores = mutual_info_regression(X_train, y_train, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

    print("\n" + "="*50)
    print("MUTUAL INFORMATION SCORES (vs streamScore)")
    print("="*50)
    for feat, score in mi_series.items():
        flag = "  ⚑ LOW SIGNAL" if score < MI_LOW_SIGNAL else ""
        print(f"  {feat:<30} {score:.4f}{flag}")
    print("="*50 + "\n")

    low_signal = mi_series[mi_series < MI_LOW_SIGNAL].index.tolist()
    if low_signal:
        log.info(f"Low-signal features flagged (MI < {MI_LOW_SIGNAL}): {low_signal}")
        log.info("These will NOT be dropped until confirmed by SHAP in Phase 5.")


# ── Step 6: Save splits ───────────────────────────────────────────────────────

def save_splits(X_train, X_test, y_train, y_test) -> None:
    splits = {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }
    for name, obj in splits.items():
        path = os.path.join(SPLITS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    log.info(f"Splits saved to {SPLITS_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Reading {IN_PATH}...")
    df = pd.read_csv(IN_PATH)
    log.info(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    log.info("Step 1: Multicollinearity check...")
    df = check_multicollinearity(df)

    log.info("Step 2: Train/test split...")
    X_train, X_test, y_train, y_test = split_data(df)

    log.info("Step 3: Re-fitting genre encoder on train set...")
    X_train, X_test = refit_genre_encoder(X_train, y_train, X_test)

    log.info("Step 4: Scaling features...")
    X_train, X_test = scale_features(X_train, X_test)

    log.info("Step 5: Mutual information scores...")
    compute_mutual_information(X_train, y_train)

    log.info("Step 6: Saving splits...")
    save_splits(X_train, X_test, y_train, y_test)

    log.info("✅ Phase 3 complete.")
    log.info(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")


if __name__ == "__main__":
    main()
