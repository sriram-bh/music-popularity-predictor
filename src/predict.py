"""
Inference: predict_success(song_profile) -> dict
=================================================
Loads the best model (XGBoost) and all transformers to predict streamScore
for a new song given its audio and metadata profile.

Usage:
    from predict import predict_success

    result = predict_success({
        "danceability":      0.735,
        "energy":            0.578,
        "key":               1,
        "loudness":          -5.4,
        "mode":              1,
        "speechiness":       0.048,
        "acousticness":      0.102,
        "instrumentalness":  0.0,
        "liveness":          0.121,
        "valence":           0.491,
        "tempo":             120.0,
        "time_signature":    4,
        "duration_ms":       215000,   # or duration_sec: 215.0
        "genre":             "pop",
        "release_month":     3,        # 1–12
        "artist_name":       "Lady Gaga",  # optional — uses global mean if unknown
    })

    # Returns:
    # {
    #     "predicted_streamscore": 187.4,
    #     "percentile":            82.1,
    #     "tier":                  "Top 20%",
    # }
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")

FEATURE_ORDER = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature", "release_month_sin",
    "release_month_cos", "duration_sec", "genre_encoded",
    "artist_avg_streamscore",
]


# ── Load transformers (cached at module level) ────────────────────────────────

def _load(filename):
    with open(os.path.join(MODELS_DIR, filename), "rb") as f:
        return pickle.load(f)


_model          = None
_scaler         = None
_genre_encoder  = None
_artist_encoder = None
_y_train        = None   # used for percentile calculation


def _ensure_loaded():
    global _model, _scaler, _genre_encoder, _artist_encoder, _y_train
    if _model is None:
        _model          = _load("xgboost.pkl")
        _scaler         = _load("scaler.pkl")
        _genre_encoder  = _load("genre_encoder.pkl")
        _artist_encoder = _load("artist_encoder.pkl")
        with open(os.path.join(SPLITS_DIR, "y_train.pkl"), "rb") as f:
            _y_train = pickle.load(f)
        log.info("Models and transformers loaded.")


# ── Feature construction ──────────────────────────────────────────────────────

def _build_features(profile: dict) -> pd.DataFrame:
    p = dict(profile)

    # duration_ms → duration_sec
    if "duration_sec" not in p:
        if "duration_ms" in p:
            p["duration_sec"] = p.pop("duration_ms") / 1000.0
        else:
            raise ValueError("Provide either 'duration_ms' or 'duration_sec'.")
    else:
        p.pop("duration_ms", None)

    # Cyclical month encoding
    month = int(p.pop("release_month", 6))
    p["release_month_sin"] = np.sin(2 * np.pi * month / 12)
    p["release_month_cos"] = np.cos(2 * np.pi * month / 12)

    # Genre encoding
    genre = p.pop("genre", None)
    gmap  = _genre_encoder["genre_map"]
    gmean = _genre_encoder["global_mean"]
    p["genre_encoded"] = gmap.get(genre, gmean) if genre else gmean
    if genre and genre not in gmap:
        log.warning(f"Unknown genre '{genre}' — using global mean ({gmean:.2f})")

    # Artist reputation
    artist = p.pop("artist_name", None)
    amap   = _artist_encoder["artist_map"]
    amean  = _artist_encoder["global_mean"]
    p["artist_avg_streamscore"] = amap.get(artist, amean) if artist else amean
    if artist and artist not in amap:
        log.warning(f"Unknown artist '{artist}' — using global mean ({amean:.2f})")

    # Build DataFrame in the exact feature order the model was trained on
    missing = [f for f in FEATURE_ORDER if f not in p]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return pd.DataFrame([{f: p[f] for f in FEATURE_ORDER}])


# ── Percentile & tier ─────────────────────────────────────────────────────────

def _percentile(score: float) -> float:
    return float((np.array(_y_train) < score).mean() * 100)


def _tier(pct: float) -> str:
    if pct >= 90:
        return "Top 10%"
    elif pct >= 75:
        return "Top 25%"
    elif pct >= 50:
        return "Above Average"
    elif pct >= 25:
        return "Below Average"
    else:
        return "Bottom 25%"


# ── Public API ────────────────────────────────────────────────────────────────

def predict_success(song_profile: dict) -> dict:
    """
    Predict streamScore for a new song.

    Parameters
    ----------
    song_profile : dict
        Required audio keys: danceability, energy, key, loudness, mode,
            speechiness, acousticness, instrumentalness, liveness, valence,
            tempo, time_signature
        Required meta keys: duration_ms (or duration_sec), release_month (1–12)
        Optional keys: genre (str), artist_name (str)

    Returns
    -------
    dict with keys:
        predicted_streamscore : float
        percentile            : float  (0–100)
        tier                  : str
    """
    _ensure_loaded()

    features_df   = _build_features(song_profile)
    features_scaled = pd.DataFrame(
        _scaler.transform(features_df),
        columns=features_df.columns,
    )
    score = float(_model.predict(features_scaled)[0])
    score = max(0.0, score)   # streamScore can't be negative

    pct  = _percentile(score)
    tier = _tier(pct)

    return {
        "predicted_streamscore": round(score, 2),
        "percentile":            round(pct, 1),
        "tier":                  tier,
    }


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = {
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
        "duration_ms":      215000,
        "genre":            "pop",
        "release_month":    3,
        "artist_name":      "Lady Gaga",
    }

    result = predict_success(demo)
    print("\n── Prediction ──────────────────────────────")
    for k, v in result.items():
        print(f"  {k:<28} {v}")
    print("────────────────────────────────────────────\n")
