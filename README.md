# Avazu CTR Prediction

Predicting click-through rate on mobile ads using the [Avazu Kaggle dataset](https://www.kaggle.com/competitions/avazu-ctr-prediction). Built entirely from scratch — no sklearn, no XGBoost. NumPy for all models and math; pandas for sampling only.

---

## Results

### Default threshold (0.5)

| Model | Log-Loss | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|----------|---------|--------|-----------|--------|----|
| Naive Bayes | 0.5493 | 0.6150 | 0.2394 | 0.326 | 0.093 | 0.144 |
| Logistic Regression (L1) | 0.4364 | 0.6466 | 0.2723 | 0.510 | 0.007 | 0.015 |
| Decision Tree | 0.4297 | 0.6657 | 0.3037 | 0.521 | 0.048 | 0.088 |
| Random Forest (10 trees) | 0.4318 | 0.6769 | 0.2980 | — | — | — |

### Optimal threshold (tuned on val set)

| Model | Threshold | Precision | Recall | F1 |
|-------|-----------|-----------|--------|----|
| Naive Bayes | 0.050 | 0.200 | 0.813 | 0.322 |
| Logistic Regression (L1) | 0.185 | 0.239 | 0.615 | 0.345 |
| Decision Tree | 0.155 | 0.263 | 0.582 | 0.363 |
| Random Forest | 0.163 | 0.242 | 0.703 | 0.360 |

### Final — Decision Tree on holdout test set (Oct 30, never touched during training)

| Log-Loss | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|----------|---------|--------|-----------|--------|----|
| 0.4293 | 0.6681 | 0.3089 | 0.273 | 0.595 | 0.374 |

**21.8% log-loss reduction** over the Naive Bayes baseline.

---

## The Story — What We Did, Why, and Every Decision

### The Problem

Avazu runs mobile ads. Every time a user sees an ad, Avazu needs to predict the probability the user will click it. That probability — the click-through rate — determines which ad to show and how much to charge the advertiser. Get it right and ad revenue goes up. Get it wrong and you're showing ads nobody clicks.

The dataset is 40 million rows of real ad impressions from October 2014, each labelled click=1 or click=0. The target is 17% positive — most impressions don't get clicked.

---

### Step 1 — Sampling: Why Not Just Load Everything?

**The problem:** 40M rows × 52 features × 4 bytes = ~8GB just for the feature matrix. A laptop has 8-16GB total RAM. Loading everything crashes the machine or makes it unusably slow.

**The naive fix:** read the first 200k rows. Fast, simple — and wrong. The Avazu data is ordered chronologically. The first 200k rows are all from October 21 at midnight. The model would never see evening hours, weekday patterns, or any behaviour from Oct 22-30.

**What we did instead — stratified sampling:**
Load all 40M rows with pandas, extract the day number from the hour column, then sample exactly 20,000 rows per day while preserving the 17% click ratio within each day. Clicks and non-clicks are sampled separately per day so the CTR is locked at 16.98% in the final sample.

Result: exactly 20,000 rows per day across all 10 days, CTR locked at 16.98% — matches the original dataset exactly.

**Why this matters:** equal daily representation means the model sees all hours, all days, and all behaviour patterns — not just Oct 21 at midnight.

---

### Step 2 — EDA: What the Data Tells Us

Before building any model, we look at the data:

- **Overall CTR: 16.98%** — the dataset is imbalanced. 83% of rows are non-clicks. A model that always predicts "no click" would be 83% accurate but completely useless.
- **device_id = a99f214a appears in 165k of 200k rows** — this is a placeholder for anonymous users (cookies blocked, first visit). Not a real device ID.
- **C20 contains -1 values** — Avazu's null marker, not a real integer.
- **Hour of day CTR varies from 15.6% to 18.7%** — time of day has signal but it's modest.
- **app_category CTR (~20%) vs site_category CTR (~13%)** — app traffic clicks more than site traffic.

---

### Step 3 — Split: Why Not Random?

**The wrong way:** shuffle all 200k rows, take 80% for train and 20% for val. Simple and standard for most datasets — but wrong here.

**Why it's wrong for time-series data:** a random split lets the model see October 29 rows during training while evaluating on October 22 rows. In production, you always predict the future from the past. A model that trains on the future leaks information — it looks better than it actually is.

**What we did — time-based split:**
- Train: Oct 21–28 (~160k rows) — the model learns on these
- Val: Oct 29 (~19k rows) — tune hyperparameters here
- Test: Oct 30 (~21k rows) — sacred holdout, evaluated once at the very end

The test set is never touched until the final step. Using it for any decision during development — even just to check a number — is data leakage.

---

### Step 4 — Feature Engineering: The Most Important Step

> *"A mediocre model with great features often beats a great model with mediocre features."*

The raw data is full of hex strings like `1fbe01fe` and integers like `15706`. A model can only do math on numbers. Feature engineering converts raw data into something a model can learn from.

**Three techniques used:**

**Frequency encoding** — for high-cardinality columns (site_id, device_id, app_id, etc.)

`site_id` has ~1,800 unique values in the sample. One-hot encoding would create 1,800 new columns — most nearly all zeros, slow to train, hard to generalise. Instead, replace each site ID with how often it appeared in training: `log(count + 1)`. A site that appears 45,000 times gets a high number; a rare site gets a low number. One column, meaningful signal.

**One-hot encoding** — for low-cardinality columns (device_type, banner_pos, C15, C16, etc.)

`device_type` has 4 unique values (0, 1, 2, 4). If you keep it as a raw integer, the model assumes device type 4 is "more" than device type 1 — a false ordering. One-hot creates one binary column per value, removing that assumption. With only 4 values it stays compact. 7 columns → 36 binary features in this sample.

**Z-score scaling** — for continuous and frequency-encoded features

Gradient descent converges much faster when all features are on the same scale. Z-score scaling subtracts the mean and divides by the standard deviation so every feature has mean 0 and standard deviation 1. Critically, the mean and standard deviation are computed on training data only — applying them to val and test avoids leakage.

**The leakage rule:** every encoding map — frequency counts, one-hot vocabularies, scaling parameters — is computed on training rows only. If we computed them on all 200k rows before splitting, val and test rows would have "seen" their own data during feature engineering. The model would appear better than it really is.

**Result:** 52 features from 24 raw columns.

---

### Step 5 — Naive Bayes: The Baseline

Before building a complex model, establish a baseline. The baseline tells you whether your improvements are real or just noise.

Naive Bayes assumes each feature independently contributes to the click probability — the "naive" assumption. It multiplies probabilities together, working in log-space to avoid floating point underflow. Laplace smoothing prevents zero probabilities for values never seen in training.

**The key weakness:** it binarizes all features at 0. A frequency-encoded site_id of 8.2 (very popular site) and 0.7 (rare site) both become 1. All that careful frequency encoding is thrown away. This is intentional — it makes the baseline weak so better models have room to improve.

**Val log-loss: 0.5493, AUC: 0.6150**

---

### Step 6 — Logistic Regression: The Workhorse

Logistic regression uses the actual feature values — not binarized. That alone is why it beats Naive Bayes.

**How it works:** computes a weighted sum of features, passes it through a sigmoid function to get a probability between 0 and 1, then adjusts weights to minimise log-loss via gradient descent.

**Mini-batch SGD:** instead of computing the gradient on all 160k rows at once (slow) or one row at a time (noisy), we use batches of 2048 rows. Fast and stable.

**Learning rate decay:** start with large steps to move fast toward the minimum, shrink the step size over time to fine-tune. `lr_t = lr / (1 + decay × epoch)`

**L1 vs L2 regularization:** both penalise large weights to prevent overfitting. L2 shrinks all weights proportionally. L1 can drive weights exactly to zero — automatic feature selection. In this project L1 and L2 tied at 0.4111, meaning most features are genuinely useful and L1 had nothing to zero out.

**Grid search:** tried 18 combinations of learning rate and regularization strength on the val set. Best: lr=0.5, lambda=0.0001, L1.

**Val log-loss: 0.4364, AUC: 0.6466 — 20.6% improvement over Naive Bayes**

**Top features by weight:**
- C18=1 pushes strongly toward no click (w=-0.73)
- C18=2 pushes toward click (w=+0.48)
- freq_site_id pushes toward click (w=+0.26) — popular sites get clicked more

---

### Step 7 — Decision Tree: Non-Linear Boundaries

Logistic Regression draws a straight line through feature space. Decision Trees ask a sequence of yes/no questions — "is the site frequency above X? is the banner position 0?" — building a tree of splits.

**Gini impurity:** at each node, we find the feature and threshold that best separates clicks from non-clicks. Gini impurity measures how mixed a node is: 0 means all one class (pure), 0.5 means perfectly mixed. We pick the split that reduces impurity the most — that's information gain.

**Regularization:** `max_depth=6` prevents the tree from growing too deep and memorising training data. `min_samples_leaf=50` ensures every leaf has at least 50 rows — no splits on tiny subgroups.

**Val log-loss: 0.4297, AUC: 0.6657 — best single model on log-loss**

---

### Step 8 — Random Forest: Ensemble Learning

One Decision Tree has high variance — change the training data slightly and you get a very different tree. Random Forest fixes this by training 10 trees and averaging their predictions.

Two sources of randomness force the trees to be different from each other:
1. **Bootstrap sampling:** each tree trains on a random sample of 160k rows drawn with replacement (~63% unique rows). Different trees see different data.
2. **Feature subsampling:** at each split, only √52 ≈ 7 features are considered, not all 52. Different trees learn from different features.

Averaging 10 decorrelated trees reduces variance without increasing bias. This is the core insight of ensemble methods.

**Val AUC: 0.6769 — highest AUC of all models**
**Val PR-AUC: 0.2980 — highest PR-AUC of all models**

---

### Step 9 — Threshold Tuning: Fixing Near-Zero Recall

At the default threshold of 0.5, Logistic Regression has 0.3% recall — it almost never predicts click=1. This isn't a model failure. It's a threshold failure.

With 17% CTR, most model outputs cluster around 0.15–0.25. Nothing crosses 0.5. Using 0.5 as the decision boundary means the model stays silent almost always.

The fix: sweep thresholds from 0.05 to 0.60 on the val set, pick the one that maximises F1. For LR this is 0.185 — close to the actual CTR of 17%, which makes intuitive sense. At this threshold recall jumps from 0.74% to 61.5%.

---

### Step 10 — Slice Analysis: Where Models Fail

Overall metrics hide subpopulation failures. We break performance down by device_type and hour of day.

Key findings:
- `device_type=0` (tablet): AUC 0.57 — models struggle here, likely too few training examples
- `device_type=5` (feature phone): only 87 val rows, 4.6% CTR — very different from the rest
- Evening hours (18–23) show slightly higher AUC than morning across all models

---

### Step 11 — Final Evaluation

The test set (Oct 30) is evaluated exactly once with the best val model (Decision Tree at threshold 0.155).

**Test results: log-loss=0.429, AUC=0.668, F1=0.374**

The val-to-test gap is tiny (0.430 → 0.429 on log-loss), which means the model generalises to new data rather than overfitting to Oct 29.

---

## Project Structure

```
avazu-ctr-prediction/
├── notebook.ipynb # full pipeline — step-by-step Jupyter notebook
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

**2. Open the notebook**
```bash
jupyter notebook notebook.ipynb
```

Run all cells top to bottom. Takes ~5–6 minutes. Most of that is loading and stratified sampling of the 40M row dataset and the LR grid search (18 model fits).

**Requirements:** Python 3.8+, NumPy only.
