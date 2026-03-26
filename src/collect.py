"""
Phase 1: Data Collection
========================
Sources:
  1. data/spotify_dataset.csv (local)
       maharshipandya/spotify-tracks-dataset
       ~114k tracks: popularity, genre, 12 audio features

  2. data/tracks.csv (local)
       yamaerenay/spotify-dataset-19212020-600k-tracks
       ~586k tracks: release_date
       Joined on track_id — expected ~23k matches

Output: data/processed/songs_raw.csv (21 columns)
"""

import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MISSING_THRESHOLD = 0.4
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


# ── Step 1: Load Spotify Tracks dataset ───────────────────────────────────────

def load_spotify_kaggle() -> pd.DataFrame:
    """
    Loads maharshipandya/spotify-tracks-dataset from local CSV.
    Provides: popularity, genre, and all 12 audio features.
    """
    path = os.path.join(DATA_DIR, "spotify_dataset.csv")
    log.info(f"Loading Spotify dataset from {path}...")
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={"artists": "artist_name", "track_genre": "genre"})
    log.info(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# ── Step 2: Load release_date from yamaerenay dataset ─────────────────────────

def load_release_dates() -> pd.DataFrame:
    """
    Loads yamaerenay/spotify-dataset-19212020-600k-tracks from local CSV.
    Only extracts id + release_date — nothing else needed.
    """
    path = os.path.join(DATA_DIR, "tracks.csv")
    log.info(f"Loading release dates from {path}...")
    df = pd.read_csv(path, usecols=["id", "release_date"])
    df = df.dropna(subset=["release_date"]).drop_duplicates("id")
    log.info(f"Loaded {len(df):,} release dates")
    return df


# ── Step 3: Merge on track_id ─────────────────────────────────────────────────

def merge_release_dates(spotify_df: pd.DataFrame, release_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join on track_id. Drops rows with no release_date match since
    release_date is required to compute streamScore in Phase 2.
    """
    before = len(spotify_df)
    release_df = release_df.rename(columns={"id": "track_id"})
    df = spotify_df.merge(release_df, on="track_id", how="inner")
    dropped = before - len(df)
    log.info(f"Merge: {before:,} → {len(df):,} rows ({dropped:,} dropped, no match)")
    return df


# ── Step 4: Drop sparse columns ───────────────────────────────────────────────

def drop_sparse_columns(df: pd.DataFrame, threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    missing_rate = df.isnull().mean()
    to_drop = missing_rate[missing_rate > threshold].index.tolist()
    if to_drop:
        log.info(f"Dropping {len(to_drop)} sparse columns (>{threshold:.0%} missing): {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        log.info("No columns exceed the missing threshold — nothing dropped.")
    return df


# ── Step 5: Audit ─────────────────────────────────────────────────────────────

def audit_dataframe(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("PHASE 1 AUDIT REPORT")
    print("="*60)
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print("\n  Missing values per column:")
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    for col, rate in missing.items():
        print(f"    {col:<40} {rate:.1%}")
    if missing.empty:
        print("    (none)")
    print("\n  Column list:")
    for c in df.columns:
        print(f"    {c}")
    print("="*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spotify_df  = load_spotify_kaggle()
    release_df  = load_release_dates()
    merged      = merge_release_dates(spotify_df, release_df)
    merged      = drop_sparse_columns(merged)

    audit_dataframe(merged)

    out_path = os.path.join(OUT_DIR, "songs_raw.csv")
    merged.to_csv(out_path, index=False)
    log.info(f"✅ Phase 1 complete. Saved to: {out_path}")


if __name__ == "__main__":
    main()
