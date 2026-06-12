# Avazu CTR Prediction

Predicting click-through rate on mobile ads using the [Avazu Kaggle dataset](https://www.kaggle.com/competitions/avazu-ctr-prediction). Built from scratch using NumPy only — no sklearn, no pandas.

Course: Jaspinder's ML Course (Days 1–20)

---

## Results

| Model | Val Log-Loss | Val AUC |
|-------|-------------|---------|
| Naive Bayes (baseline) | 0.5389 | 0.6253 |
| Logistic Regression (val) | 0.4068 | 0.6721 |
| Logistic Regression (test) | 0.4218 | 0.6722 |

Logistic Regression achieves a **24.5% reduction in log-loss** over the Naive Bayes baseline.

---

## Dataset

- **Source:** Avazu click log, 10 days of mobile ad impressions (Oct 21–30, 2014)
- **Size:** ~40M rows, 5.9 GB
- **Target:** `click` — binary (1 = clicked, 0 = not clicked)
- **Class imbalance:** ~17% positive (clicked)
- **Features:** Mix of anonymized categorical IDs (site, app, device) and ad slot attributes

---

## Approach

### Sampling
Loading all 40M rows into memory (~8 GB as float32) is impractical on a laptop. Instead we use **reservoir sampling** to draw 200k rows uniformly at random from the full file, streaming one row at a time. This gives ~20k rows per day across all 10 days.

### Split
Data is split by **calendar date**, not randomly. A random split would leak future rows into training — the model would see Oct 29 data while "predicting" Oct 22. Time-based split matches how a real CTR system works: always train on the past, predict the future.

| Split | Dates | Rows |
|-------|-------|------|
| Train | Oct 21–28 | ~160k |
| Val | Oct 29 | ~19k |
| Test (holdout) | Oct 30 | ~21k |

### Feature Engineering
All features are numeric — the model can't work with raw hex strings or category labels.

- **Frequency encoding** — high-cardinality columns (`site_id`, `device_id`, etc.): replace each value with `log(count + 1)` from the training set. Captures popularity without exploding column count.
- **One-hot encoding** — low-cardinality columns (`device_type`, `banner_pos`, etc.): one binary column per unique value. Removes false ordering that raw integers imply.
- **Z-score scaling** — applied to frequency and temporal features. Fit on train only, applied to val/test.

Encoding maps are **fit on training data only** — applying them to val/test before fitting would leak information about those splits.

### Models

**Naive Bayes (baseline)**
Bernoulli Naive Bayes assumes each feature independently contributes to the click probability. Fast to train, interpretable, but binarizes all feature values at 0 — discarding the magnitude information from frequency encoding.

**Logistic Regression**
Mini-batch SGD with L2 regularization. Uses the actual feature values (not binarized), which is the main reason it outperforms Naive Bayes here. Hyperparameters (learning rate, L2 lambda) are tuned via grid search on the val set. Final model is retrained on train + val combined before evaluating on the holdout test set.

---

## Project Structure

```
avazu-ctr-prediction/
├── main.py        # full pipeline: load → EDA → split → features → NB → LR → eval
├── train          # raw Avazu training data (not in repo — download from Kaggle)
├── test           # raw Avazu test data (not in repo — download from Kaggle)
└── README.md
```

---

## How to Run

**1. Download the data from Kaggle**

```bash
# From https://www.kaggle.com/competitions/avazu-ctr-prediction/data
# Place 'train' and 'test' files in the project directory
```

**2. Run the pipeline**

```bash
python3 main.py
```

Takes ~3–4 minutes (most of that is streaming 40M rows for reservoir sampling). Prints EDA, baseline results, grid search table, and final test evaluation.

**Requirements:** Python 3.8+, NumPy only.

---

## Key Concepts

| Concept | Where it appears |
|---------|-----------------|
| Reservoir sampling | `reservoir_sample()` — memory-safe uniform sampling from large files |
| Time-based train/val/test split | `time_based_split()` — prevents temporal leakage |
| Frequency encoding | `FeatureEngineer.FREQ_COLS` — converts high-cardinality categoricals to numeric |
| One-hot encoding | `FeatureEngineer.ONEHOT_COLS` — converts low-cardinality categoricals without false ordering |
| Laplace smoothing | `NaiveBayes.__init__(alpha=1.0)` — prevents zero probabilities for unseen values |
| Log-space arithmetic | `NaiveBayes.predict_proba()` — avoids floating point underflow |
| Mini-batch SGD | `LogisticRegression.fit()` — faster than full-batch, more stable than single-sample |
| L2 regularization | `LogisticRegression` — penalizes large weights to reduce overfitting |
| Learning rate decay | `LogisticRegression.fit()` — large steps early, fine-tune near minimum |
| Holdout discipline | Test set evaluated exactly once at the very end |
