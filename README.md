# Song Success Prediction Engine

A machine learning pipeline that predicts the commercial success of a song given its audio and metadata profile. Trained on 23,342 Spotify tracks across five phases: data collection, feature engineering, preprocessing, model training, and evaluation.

---

## Key Findings

**The single biggest finding: genre dominates everything.** It's the #1 feature by a massive margin across all three models and both SHAP plots. Before a note is heard, your genre determines your streamScore ceiling.

**Audio features — what artists can actually control:**

- **Loudness** is the most consistent signal with the clearest direction. Master to -6 to -8 LUFS integrated. Acoustic and folk artists frequently leave 4-6 dB on the table — every genre benefits from hitting this range.

- **Valence** effects are asymmetric. Euphoric, maximally upbeat songs get a meaningful boost; the penalty for low valence is milder. The advice isn't "avoid sad songs" but "if you're going upbeat, commit fully." Half-happy doesn't capture the uplift bonus.

- **Liveness** is a clean negative — keep it below 0.2. Audience noise and room ambience actively hurt scores. Record in a treated space.

- **Danceability** and **acousticness** are genre-dependent. Don't maximize blindly — match your genre's norms. A genre-stratified SHAP interaction plot would give precise per-genre guidance here; worth generating.

- **Features that don't matter at all:** key, mode, time signature, and release month. All effectively zero across every model. Don't make creative or scheduling decisions based on these.

**The artist reputation problem is real and large.** Adding a single artist feature jumps XGBoost R² from 0.51 to 0.74 — one feature outweighs all 15 audio features combined. The honest read: ~50% of a song's success is predictable from the song itself. Most of the rest is who made it. The no-artist model is the only version that tells emerging artists anything useful.

**Which model to trust:** XGBoost for predictions (R²=0.51), Lasso for confirming which features are genuinely zero (R²=0.35). Both tree models show mild overfitting — manageable at 23k rows, revisit with more data.

**The honest ceiling:** With artist reputation included, XGBoost explains 74% of variance — meaning only 26% is genuinely unexplained noise. Without it, that rises to 49%. The 26% gap between those two numbers is the quantified weight of artist reputation. The remaining 26% that no version of the model can explain is playlist placement, algorithmic amplification, marketing spend, and virality — structural factors invisible to any audio-based model. This predicts intrinsic potential, not industry outcomes.

---

## Project Structure

```
music_popularity_predictor/
├── src/
│   ├── collect.py                  # Phase 1: data collection and merging
│   ├── engineer.py                 # Phase 2: feature engineering (with artist)
│   ├── engineer_no_artist.py       # Phase 2: feature engineering (no artist)
│   ├── preprocess.py               # Phase 3: preprocessing and splits (with artist)
│   ├── preprocess_no_artist.py     # Phase 3: preprocessing and splits (no artist)
│   ├── train.py                    # Phase 4: model training (with artist)
│   ├── train_no_artist.py          # Phase 4: model training (no artist)
│   ├── evaluate.py                 # Phase 5: evaluation and plots (with artist)
│   ├── evaluate_no_artist.py       # Phase 5: evaluation and plots (no artist)
│   ├── predict_all.py              # Predict a new song across all 3 models
│   └── predict.py                  # Single-model inference function
├── data/
│   ├── spotify_dataset.csv         # Source 1: maharshipandya/spotify-tracks-dataset
│   ├── tracks.csv                  # Source 2: yamaerenay/spotify-dataset-19212020
│   ├── processed/                  # Intermediate CSVs
│   └── splits/                     # Train/test pkl splits
├── models_with_artist/             # Trained models including artist reputation
├── models_no_artist/               # Trained models — audio + genre only
├── outputs_with_artist/            # Plots and reports (with artist)
├── outputs_no_artist/              # Plots and reports (no artist)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Mac users:** XGBoost requires OpenMP. Install with `brew install libomp` if you hit a `libxgboost.dylib` error.

### 2. Add data

Download the two source datasets and place them in `data/`:

- [maharshipandya/spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) → `data/spotify_dataset.csv`
- [yamaerenay/spotify-dataset-19212020-600k-tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks) → `data/tracks.csv`

### 3. Run the pipeline

```bash
# With artist reputation
python src/collect.py
python src/engineer.py
python src/preprocess.py
python src/train.py
python src/evaluate.py

# Without artist reputation (for emerging artist predictions)
python src/engineer_no_artist.py
python src/preprocess_no_artist.py
python src/train_no_artist.py
python src/evaluate_no_artist.py
```

### 4. Predict a new song

Edit the `SONG` dict at the top of `src/predict_all.py` and run:

```bash
python src/predict_all.py
```

Output:
```
====================================================
  SONG SUCCESS PREDICTION
====================================================
  Model              Score     Pct   Tier
----------------------------------------------------
  Lasso             153.20   62.4%  Above Average
  Random Forest     171.40   74.1%  Top 25%
  XGBoost           168.90   73.2%  Top 25%
====================================================
  Ensemble (avg)    164.50   70.2%  Top 25%
====================================================
```

---

## Target Variable

```
streamScore = popularity × (1 + log10(1 + years_since_release))^1.2
```

Spotify's `popularity` score (0–100) reflects recent stream counts and saves, heavily favouring new releases. The multiplier adjusts upward for older songs with sustained relevance. The exponent 1.2 rewards longevity without unfairly penalising newer songs that achieved equivalent streams in less time.

Range in this dataset: **0–287**, mean ~145, std ~41.

---

## Models

| Model | No-Artist Test R² | With-Artist Test R² |
|---|---|---|
| Lasso | 0.352 | 0.663 |
| Random Forest | 0.484 | 0.741 |
| XGBoost | 0.510 | 0.738 |

Three models are trained in each pipeline variant:

- **Lasso** — linear baseline with L1 regularisation; alpha selected via 5-fold CV. Best for interpretability and identifying truly zero-signal features.
- **Random Forest** — 500 trees, hyperparameters tuned via 5-fold CV. Strong generalisation.
- **XGBoost** — gradient boosted trees, 500 estimators, hyperparameters tuned via 5-fold CV. Best overall test performance.

---

## The Artist vs. No-Artist Decision

Two pipeline variants were built deliberately:

**With artist** (`engineer.py` → `models_with_artist/`) includes `artist_avg_streamscore` — a leave-one-out mean streamScore of all other songs by the same artist in the dataset. This captures existing artist reputation but has two problems:
1. Circular for small artists (2–3 songs in the dataset — the mean essentially reflects the artist's own scores)
2. Useless for emerging artists — it tells you "popular artists make popular songs," which is not actionable

**Without artist** (`engineer_no_artist.py` → `models_no_artist/`) uses only audio features, genre, duration, and release timing. This is the version that answers the question this engine was built to answer: *does this song have the intrinsic profile to succeed?*

---

## Data Sources

| Dataset | Source | Rows | Key columns |
|---|---|---|---|
| spotify-tracks-dataset | maharshipandya (Kaggle) | 114k | popularity, genre, 12 audio features |
| spotify-dataset-19212020 | yamaerenay (Kaggle) | 600k | track_id, release_date |

Inner join on `track_id` → **23,342 matched rows**.
