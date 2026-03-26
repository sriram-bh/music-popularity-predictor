"""
Phase 2: Feature Engineering
=============================
Input:  data/processed/songs_raw.csv
Output: data/processed/songs_features.csv
        models/genre_encoder.pkl

Transformations applied:
  1. streamScore       — target: popularity × (1 + log10(1 + years_since_release))²
  2. release_month     — cyclical sin/cos encoding
  3. duration_ms       — convert to duration_sec
  4. genre             — target encoding (mean streamScore per genre)
                         saved to models/genre_encoder.pkl for inference
  5. Drop unused cols  — metadata, raw date fields, popularity, genre string

Audio features (danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, time_signature)
are kept as-is. Phase 3 handles multicollinearity checks and scaling.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
IN_PATH    = os.path.join(BASE_DIR, "data", "processed", "songs_raw.csv")
OUT_PATH   = os.path.join(BASE_DIR, "data", "processed", "songs_features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Step 1: Compute streamScore ───────────────────────────────────────────────

def compute_stream_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    streamScore = popularity × (1 + log10(1 + years_since_release))^1.2

    Balances longevity and recency. Old songs with sustained popularity are
    rewarded, but a newer song achieving equal popularity in less time is
    treated as comparably impressive. Exponent 1.2 keeps the age bonus
    meaningful without overwhelming recent breakout hits.
    """
    df["release_date_parsed"] = pd.to_datetime(df["release_date"], errors="coerce")

    # Fallback: year-only strings like "1999"
    mask = df["release_date_parsed"].isna()
    if mask.any():
        df.loc[mask, "release_date_parsed"] = pd.to_datetime(
            df.loc[mask, "release_date"].astype(str).str[:4],
            format="%Y", errors="coerce"
        )

    now = pd.Timestamp.now()
    years = ((now - df["release_date_parsed"]).dt.days / 365.25).clip(lower=0)
    df["years_since_release"] = years
    df["streamScore"] = df["popularity"] * (1 + np.log10(1 + years)) ** 1.2

    n_null = df["streamScore"].isna().sum()
    if n_null:
        log.warning(f"streamScore is NaN for {n_null:,} rows — dropping them")
        df = df.dropna(subset=["streamScore"])

    log.info(f"streamScore — min: {df['streamScore'].min():.2f}, "
             f"max: {df['streamScore'].max():.2f}, "
             f"mean: {df['streamScore'].mean():.2f}")
    return df


# ── Step 2: Cyclical release_month encoding ───────────────────────────────────

def encode_release_month(df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos encode month so December and January are treated as adjacent."""
    df["release_month"] = df["release_date_parsed"].dt.month.fillna(6).astype(int)
    df["release_month_sin"] = np.sin(2 * np.pi * df["release_month"] / 12)
    df["release_month_cos"] = np.cos(2 * np.pi * df["release_month"] / 12)
    return df


# ── Step 3: duration_ms → duration_sec ───────────────────────────────────────

def convert_duration(df: pd.DataFrame) -> pd.DataFrame:
    if "duration_ms" in df.columns:
        df["duration_sec"] = df["duration_ms"] / 1000.0
        df = df.drop(columns=["duration_ms"])
    return df


# ── Step 4: Target encode genre ───────────────────────────────────────────────

def encode_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target encoding: replace genre string with mean streamScore per genre.

    NOTE: fitted on full dataset here for inspection and inference use.
    Phase 3 re-fits this encoder on X_train only to prevent leakage during
    model evaluation.
    """
    if "genre" not in df.columns:
        log.warning("No 'genre' column — skipping genre encoding.")
        return df

    genre_map = df.groupby("genre")["streamScore"].mean().to_dict()
    global_mean = df["streamScore"].mean()
    df["genre_encoded"] = df["genre"].map(genre_map).fillna(global_mean)

    encoder_path = os.path.join(MODELS_DIR, "genre_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump({"genre_map": genre_map, "global_mean": global_mean}, f)
    log.info(f"Genre encoder saved ({len(genre_map)} genres) → {encoder_path}")
    return df


# ── Step 5: Normalize enrichment features (if present) ───────────────────────

def normalize_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalize enrichment columns to [0, 1] if present.
    Skipped silently if columns are absent (they were dropped in Phase 1
    due to >40% missing when Spotify API credentials were unavailable).
    """
    cols = ["artist_followers", "album_api_popularity", "artist_api_popularity"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    scaler = MinMaxScaler()
    df[present] = scaler.fit_transform(df[present])
    log.info(f"Normalized enrichment features: {present}")
    return df


# ── Step 6: Artist reputation feature ────────────────────────────────────────

def compute_artist_avg_streamscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-out mean streamScore per artist.

    For each song, computes the mean streamScore of all OTHER songs by the same
    artist. This captures artist tier (established vs unknown) without the song
    influencing its own feature value.

    Artists with only one song in the dataset fall back to the global mean —
    they're unknowns, so average expectation is the right prior.
    """
    if "artist_name" not in df.columns:
        log.warning("No 'artist_name' column — skipping artist reputation feature.")
        return df

    artist_sum   = df.groupby("artist_name")["streamScore"].transform("sum")
    artist_count = df.groupby("artist_name")["streamScore"].transform("count")

    # Exclude the song's own streamScore from the mean
    loo_mean = (artist_sum - df["streamScore"]) / (artist_count - 1)

    # Solo artists (count=1) → NaN → fall back to global mean
    global_mean = df["streamScore"].mean()
    df["artist_avg_streamscore"] = loo_mean.fillna(global_mean)

    # Save artist lookup for inference (full mean per artist, not leave-one-out)
    artist_map = df.groupby("artist_name")["streamScore"].mean().to_dict()
    artist_encoder_path = os.path.join(MODELS_DIR, "artist_encoder.pkl")
    with open(artist_encoder_path, "wb") as f:
        pickle.dump({"artist_map": artist_map, "global_mean": global_mean}, f)
    log.info(f"Artist encoder saved ({len(artist_map)} artists) → {artist_encoder_path}")

    log.info(
        f"artist_avg_streamscore — min: {df['artist_avg_streamscore'].min():.2f}, "
        f"max: {df['artist_avg_streamscore'].max():.2f}, "
        f"mean: {df['artist_avg_streamscore'].mean():.2f}"
    )
    return df


# ── Step 7: Drop unused columns ───────────────────────────────────────────────

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [
        "track_id", "track_name", "artist_name", "album_name",
        "release_date", "release_date_parsed", "release_month",
        "years_since_release", "popularity", "genre", "explicit",
    ]
    existing = [c for c in drop if c in df.columns]
    df = df.drop(columns=existing)
    log.info(f"Dropped {len(existing)} unused columns")
    return df


# ── Audit ─────────────────────────────────────────────────────────────────────

def audit(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("PHASE 2 AUDIT REPORT")
    print("="*60)
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"\n  streamScore distribution:")
    print(df["streamScore"].describe().to_string())
    print(f"\n  Missing values:")
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    for col, rate in missing.items():
        print(f"    {col:<40} {rate:.1%}")
    if missing.empty:
        print("    (none)")
    print(f"\n  Columns:")
    for c in df.columns:
        print(f"    {c}")
    print("="*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Reading {IN_PATH}...")
    df = pd.read_csv(IN_PATH)
    log.info(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    log.info("Step 1: Computing streamScore...")
    df = compute_stream_score(df)

    log.info("Step 2: Encoding release_month...")
    df = encode_release_month(df)

    log.info("Step 3: Converting duration_ms → duration_sec...")
    df = convert_duration(df)

    log.info("Step 4: Target encoding genre...")
    df = encode_genre(df)

    log.info("Step 5: Normalizing enrichment features...")
    df = normalize_continuous(df)

    log.info("Step 6: Computing artist reputation feature...")
    df = compute_artist_avg_streamscore(df)

    log.info("Step 7: Dropping unused columns...")
    df = drop_unused_columns(df)

    audit(df)

    df.to_csv(OUT_PATH, index=False)
    log.info(f"✅ Phase 2 complete. Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
