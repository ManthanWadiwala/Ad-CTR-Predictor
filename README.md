# Avazu CTR Prediction

Predicting click-through rate on mobile ads using the [Avazu Kaggle dataset](https://www.kaggle.com/competitions/avazu-ctr-prediction). Built entirely from scratch using NumPy only — no sklearn, no pandas, no XGBoost.

Course: Jaspinder's ML Course (Days 1–28)

---

## Results

### Default threshold (0.5)

| Model | Log-Loss | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|----------|---------|--------|-----------|--------|----|
| Naive Bayes | 0.5389 | 0.6253 | 0.2459 | 0.321 | 0.134 | 0.189 |
| Logistic Regression (L1) | 0.4111 | 0.6608 | 0.2790 | 0.435 | 0.003 | 0.007 |
| Decision Tree | 0.4070 | 0.6714 | 0.2852 | 0.491 | 0.084 | 0.143 |

### Optimal threshold (tuned on val set)

| Model | Threshold | Precision | Recall | F1 |
|-------|-----------|-----------|--------|----|
| Naive Bayes | 0.174 | 0.198 | 0.653 | 0.304 |
| Logistic Regression (L1) | 0.196 | 0.250 | 0.528 | 0.340 |
| Decision Tree | 0.163 | 0.248 | 0.562 | 0.345 |

### Final — Logistic Regression on holdout test set (Oct 30, never seen during training)

| Log-Loss | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|----------|---------|--------|-----------|--------|----|
| 0.4220 | 0.6712 | 0.2960 | 0.262 | 0.473 | 0.337 |

**24.5% log-loss reduction** over the Naive Bayes baseline.

---

## Dataset

- **Source:** Avazu click log, 10 days of mobile ad impressions (Oct 21–30, 2014)
- **Size:** ~40M rows, 5.9 GB
- **Target:** `click` — binary (1 = clicked, 0 = not clicked)
- **Class imbalance:** ~17% positive (clicked)
- **Features:** Anonymized categorical IDs (site, app, device) and ad slot attributes (banner position, dimensions)

---

## Pipeline

### 1. Reservoir Sampling
Loading all 40M rows (~8 GB as float32) is impractical on a laptop. Instead we stream through all 40M rows and keep 200k sampled uniformly at random using Vitter's Algorithm R. This gives ~20k rows per day across all 10 days — verified in the EDA date distribution check.

Taking the first 200k rows sequentially would give only Oct 21, meaning the model never sees evening hours, weekday patterns, or Oct 29/30 behaviour.

### 2. Time-Based Split
Split by calendar date, not randomly. A random split leaks future rows into training — the model would see Oct 29 data while "predicting" Oct 22.

| Split | Dates | Rows |
|-------|-------|------|
| Train | Oct 21–28 | ~160k |
| Val | Oct 29 | ~19k |
| Test (holdout) | Oct 30 | ~21k |

### 3. Feature Engineering
All features are numeric — the model can't work with raw hex strings.

- **Frequency encoding** — high-cardinality columns (`site_id`, `device_id`, etc.): replace each value with `log(count + 1)` from training. Captures popularity signal in one number without exploding column count.
- **One-hot encoding** — low-cardinality columns (`device_type`, `banner_pos`, etc.): one binary column per unique value. Removes false ordering that raw integers imply.
- **Z-score scaling** — frequency and temporal features only. Fit on train, applied to val/test — no leakage.

Encoding maps fit on training data only. Applying to val/test before fitting would leak information.

### 4. Models

**Naive Bayes (baseline)**
Bernoulli Naive Bayes in log-space with Laplace smoothing. Intentionally weak — binarizes all features at 0, throwing away the magnitude information from frequency encoding. Sets the floor for other models to beat.

**Logistic Regression**
Mini-batch SGD with learning rate decay. Grid search over learning rate, L2 lambda, and L1 lambda (18 combinations). L1 and L2 tied at 0.4111 — most features are genuinely useful so L1 doesn't need to zero anything out.

**Decision Tree**
Built from scratch using Gini impurity. Splits search all 59 features × 20 sampled thresholds per node. Regularized via `max_depth=6` and `min_samples_leaf=50`. Slightly outperforms LR on log-loss (0.407 vs 0.411) and AUC (0.671 vs 0.661).

### 5. Threshold Tuning
The default threshold of 0.5 nearly eliminates recall — with 17% CTR, most model outputs are in the 0.15–0.25 range, so almost nothing crosses 0.5. The optimal threshold is found by sweeping 0.05–0.60 on the val set and picking the value that maximises F1. For LR this is 0.196 (close to the actual CTR of 17%), which recovers recall from 0.3% to 52.8%.

### 6. Slice Analysis
Performance broken down by `device_type` and hour group to identify where models are strong or weak:
- `device_type=5` (87 rows, 4.6% CTR) behaves very differently from `device_type=1` (17k rows, 15.6% CTR)
- Evening hours (18–23) show slightly higher AUC than morning hours across both models

---

## Key Concepts Covered

| Concept | Day | Where in code |
|---------|-----|---------------|
| Generators & large file handling | 1 | `reservoir_sample()` |
| NumPy vectorization | 2 | Throughout |
| Numerical stability (`log1p`, `logaddexp`, `clip`) | 4 | NB, LR sigmoid, freq encoding |
| EDA — distributions, class imbalance, leakage | 5 | `run_eda()` |
| Gradients & chain rule | 6 | `LogisticRegression.fit()` |
| Bayes' theorem, log-space arithmetic | 7 | `NaiveBayes` |
| Logistic regression & cross-entropy | 10 | `LogisticRegression` |
| Naive Bayes & Laplace smoothing | 14 | `NaiveBayes` |
| Decision Trees & Gini impurity | 15 | `DecisionTree` |
| Precision, Recall, F1, Confusion Matrix | 12/21 | `print_eval()` |
| ROC-AUC & PR-AUC | 12/21 | `roc_auc()`, `pr_auc()` |
| L1 vs L2 regularization | 22 | `LogisticRegression(reg=)` |
| Learning curves (train vs val loss) | 22 | `print_learning_curve()` |
| Threshold optimization | 12/21 | `find_optimal_threshold()` |
| Time-based split & leakage prevention | 18/26 | `time_based_split()` |
| Feature importance via weights | 20/23 | `top_features()` |
| Slice analysis / error analysis | 23 | `slice_analysis()` |
| Holdout discipline | 19/26 | Test set touched once |
| Frequency encoding | 20 | `FeatureEngineer.FREQ_COLS` |
| One-hot encoding | 20 | `FeatureEngineer.ONEHOT_COLS` |
| Z-score scaling (fit on train only) | 20 | `FeatureEngineer.fit_transform()` |

---

## Project Structure

```
avazu-ctr-prediction/
├── main.py        # full pipeline — single file, NumPy only
├── train          # raw Avazu data (not in repo — download from Kaggle)
├── test           # raw Avazu test data (not in repo)
└── README.md
```

---

## How to Run

**1. Download data from Kaggle**
```bash
# https://www.kaggle.com/competitions/avazu-ctr-prediction/data
# Place 'train' file in the project directory
```

**2. Run the pipeline**
```bash
python3 main.py
```

Takes ~5–6 minutes. Most of that is reservoir sampling (streaming 40M rows) and the LR grid search (18 model fits).

**Requirements:** Python 3.8+, NumPy only.
