"""
Predict streamScore for a new song across all 3 models (with artist).
Edit the SONG dict below and run:  python src/predict_all.py
"""

import os
import pickle
import numpy as np
import pandas as pd

# ── Edit this ─────────────────────────────────────────────────────────────────

SONG = {
    "danceability":     0.735,
    "energy":           0.578,
    "key":              1,
    "loudness":         -5.4,
    "mode":             1,
    "speechiness":      0.048,
    "acousticness":     0.102,
    "instrumentalness": 0.0,
    "liveness":         0.121,
    "valence":          0.491,
    "tempo":            120.0,
    "time_signature":   4,
    "duration_ms":      215000,    # milliseconds
    "genre":            "pop",
    "release_month":    3,         # 1–12
    "artist_name":      "Lady Gaga",  # leave blank "" if unknown
}

# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models_with_artist")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")

FEATURE_ORDER = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature", "release_month_sin",
    "release_month_cos", "duration_sec", "genre_encoded",
    "artist_avg_streamscore",
]


def load(filename):
    with open(os.path.join(MODELS_DIR, filename), "rb") as f:
        return pickle.load(f)


def build_features(song):
    p = dict(song)

    # duration
    p["duration_sec"] = p.pop("duration_ms", 0) / 1000.0

    # cyclical month
    month = int(p.pop("release_month", 6))
    p["release_month_sin"] = np.sin(2 * np.pi * month / 12)
    p["release_month_cos"] = np.cos(2 * np.pi * month / 12)

    # genre encoding
    genre_enc  = load("genre_encoder.pkl")
    genre      = p.pop("genre", None)
    p["genre_encoded"] = genre_enc["genre_map"].get(genre, genre_enc["global_mean"])

    # artist encoding
    artist_enc = load("artist_encoder.pkl")
    artist     = p.pop("artist_name", None)
    p["artist_avg_streamscore"] = artist_enc["artist_map"].get(artist, artist_enc["global_mean"])
    if artist and artist not in artist_enc["artist_map"]:
        print(f"  Note: '{artist}' not in training data — using global mean for artist reputation.")

    return pd.DataFrame([{f: p[f] for f in FEATURE_ORDER}])


def percentile(score, y_train):
    return float((np.array(y_train) < score).mean() * 100)


def tier(pct):
    if pct >= 90: return "Top 10%"
    if pct >= 75: return "Top 25%"
    if pct >= 50: return "Above Average"
    if pct >= 25: return "Below Average"
    return "Bottom 25%"


def main():
    scaler  = load("scaler.pkl")
    lasso   = load("lasso.pkl")
    rf      = load("random_forest.pkl")
    xgb     = load("xgboost.pkl")

    with open(os.path.join(SPLITS_DIR, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)

    features        = build_features(SONG)
    features_scaled = pd.DataFrame(
        scaler.transform(features), columns=features.columns
    )

    scores = {
        "Lasso":         max(0.0, float(lasso.predict(features_scaled)[0])),
        "Random Forest": max(0.0, float(rf.predict(features_scaled)[0])),
        "XGBoost":       max(0.0, float(xgb.predict(features_scaled)[0])),
    }

    print("\n" + "="*52)
    print("  SONG SUCCESS PREDICTION")
    print("="*52)
    print(f"  {'Model':<18} {'Score':>8}  {'Pct':>6}  {'Tier'}")
    print("-"*52)
    for model_name, score in scores.items():
        pct = percentile(score, y_train)
        print(f"  {model_name:<18} {score:>8.2f}  {pct:>5.1f}%  {tier(pct)}")
    print("="*52)

    avg = np.mean(list(scores.values()))
    pct = percentile(avg, y_train)
    print(f"  {'Ensemble (avg)':<18} {avg:>8.2f}  {pct:>5.1f}%  {tier(pct)}")
    print("="*52 + "\n")


if __name__ == "__main__":
    main()
