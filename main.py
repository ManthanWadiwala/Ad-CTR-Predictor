"""
Avazu CTR Prediction
====================
Course: Jaspinder's ML Course (Days 1-20)

Pipeline:
  1. Reservoir sample 200k rows from 40M  (memory-safe, spans all 10 days)
  2. EDA — click rate, column distributions, CTR by hour
  3. Time-based split — train Oct 21-28 / val Oct 29 / test Oct 30
  4. Feature engineering — frequency encode, one-hot encode, z-score scale
  5. Naive Bayes baseline  (from scratch, NumPy only)
  6. Logistic Regression   (mini-batch SGD, L2 regularization, grid search)
  7. Final evaluation on holdout test set
"""

import csv
import os
import numpy as np
from collections import Counter


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def reservoir_sample(filepath, k=200_000, seed=42):
    """
    Stream all rows and return k rows sampled uniformly at random.

    Taking the first 200k rows sequentially gives only Oct 21 — the model
    never sees later dates or peak hours. Reservoir sampling guarantees every
    row has an equal k/n chance of selection, so the sample spans all 10 days.

    Algorithm (Vitter's Algorithm R):
      Fill reservoir with first k rows. For each subsequent row i, draw
      j in [0, i]. If j < k, replace reservoir[j] with the new row.
      Every row ends up in the reservoir with probability k/n.
    """
    rng = np.random.default_rng(seed)
    reservoir = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < k:
                reservoir.append(row)
            else:
                j = int(rng.integers(0, i + 1))
                if j < k:
                    reservoir[j] = row
            if (i + 1) % 5_000_000 == 0:
                print(f"  ... streamed {(i+1)//1_000_000}M rows")
    print(f"  Sampled {len(reservoir):,} rows from {i+1:,} total")
    return reservoir


# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────

def run_eda(rows):
    print("\n" + "="*60)
    print("EDA REPORT")
    print("="*60)

    n      = len(rows)
    clicks = sum(int(r['click']) for r in rows)
    print(f"\nTotal rows   : {n:,}")
    print(f"Total clicks : {clicks:,}")
    print(f"Overall CTR  : {100*clicks/n:.2f}%")

    cat_cols = [
        'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
        'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip',
        'device_model', 'device_type', 'device_conn_type',
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]

    print(f"\n{'Column':<20} {'Unique':>8}  {'Top value':>15}  {'Rows':>8}  {'CTR':>6}")
    print("-" * 68)
    for col in cat_cols:
        counter = Counter()
        ctr_map = {}
        for r in rows:
            v = r[col]
            counter[v] += 1
            if v not in ctr_map:
                ctr_map[v] = [0, 0]
            ctr_map[v][0] += int(r['click'])
            ctr_map[v][1] += 1
        top_val, top_cnt = counter.most_common(1)[0]
        top_ctr = 100 * ctr_map[top_val][0] / ctr_map[top_val][1]
        print(f"{col:<20} {len(counter):>8}  {top_val:>15}  {top_cnt:>8}  {top_ctr:>5.1f}%")

    print("\n--- CTR by hour of day ---")
    hour_stats = {}
    for r in rows:
        h = int(r['hour']) % 100
        if h not in hour_stats:
            hour_stats[h] = [0, 0]
        hour_stats[h][0] += int(r['click'])
        hour_stats[h][1] += 1
    for h in sorted(hour_stats):
        c, t = hour_stats[h]
        bar = '#' * int(50 * c / t)
        print(f"  Hour {h:02d}: CTR={100*c/t:5.1f}%  {bar}")

    print("\n--- Date distribution (reservoir check) ---")
    day_counts = Counter((int(r['hour']) // 100) % 100 for r in rows)
    for day in sorted(day_counts):
        print(f"  Oct {day}: {day_counts[day]:,} rows")


# ─────────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────────

def time_based_split(rows):
    """
    Split by date to avoid leakage — model always trains on the past,
    evaluates on the future, matching how a real CTR system would work.

      Train : Oct 21-28  (learns patterns)
      Val   : Oct 29     (tune hyperparameters)
      Test  : Oct 30     (holdout — evaluated once at the very end)
    """
    train_rows, val_rows, test_rows = [], [], []
    for r in rows:
        day = (int(r['hour']) // 100) % 100
        if day <= 28:
            train_rows.append(r)
        elif day == 29:
            val_rows.append(r)
        else:
            test_rows.append(r)

    def ctr(split):
        return 100 * sum(r['click'] == '1' for r in split) / len(split) if split else 0.0

    print(f"  Train (Oct 21-28): {len(train_rows):,} rows  CTR={ctr(train_rows):.2f}%")
    print(f"  Val   (Oct 29)   : {len(val_rows):,} rows  CTR={ctr(val_rows):.2f}%")
    print(f"  Test  (Oct 30)   : {len(test_rows):,} rows  CTR={ctr(test_rows):.2f}%  <- holdout")
    return train_rows, val_rows, test_rows


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

class FeatureEngineer:
    """
    Converts raw row dicts into a numeric matrix.

    Frequency encoding  — high-cardinality columns (site_id, device_id, etc.)
      Replaces each value with log(count + 1) from the training set.
      Captures popularity signal in one number without exploding column count.

    One-hot encoding  — low-cardinality columns (device_type, banner_pos, etc.)
      Creates one binary column per unique value.
      Removes false ordering that raw integers would imply.

    Z-score scaling  — applied to frequency + temporal features only.
      Fit on train, applied to val/test — no leakage.
      Keeps all features on the same scale for gradient descent.
    """

    FREQ_COLS = [
        'site_id', 'site_domain', 'site_category',
        'app_id', 'app_domain', 'app_category',
        'device_id', 'device_ip', 'device_model',
        'C14', 'C17', 'C19', 'C20', 'C21'
    ]
    ONEHOT_COLS = ['C1', 'C18', 'device_type', 'device_conn_type', 'banner_pos', 'C15', 'C16']

    def __init__(self):
        self.freq_maps   = {}
        self.onehot_maps = {}
        self.scale_params = {}
        self.feature_names = []
        self.fitted = False

    def fit(self, rows):
        for col in self.FREQ_COLS:
            self.freq_maps[col] = Counter(r[col] for r in rows)
        for col in self.ONEHOT_COLS:
            self.onehot_maps[col] = sorted(set(r[col] for r in rows))
        self.fitted = True

    def _row_to_features(self, row):
        feats = []
        h_raw       = int(row['hour'])
        hour_of_day = h_raw % 100
        day_of_week = (h_raw // 100) % 100 % 7
        feats.append(float(hour_of_day))
        feats.append(float(day_of_week))
        for col in self.FREQ_COLS:
            feats.append(np.log1p(self.freq_maps[col].get(row[col], 0)))
        for col in self.ONEHOT_COLS:
            v = row[col]
            for cat in self.onehot_maps[col]:
                feats.append(1.0 if v == cat else 0.0)
        return feats

    def transform(self, rows):
        assert self.fitted, "Call fit() before transform()"
        X = np.array([self._row_to_features(r) for r in rows], dtype=np.float32)
        if not self.feature_names:
            self.feature_names = ['hour_of_day', 'day_of_week']
            self.feature_names += [f'freq_{c}' for c in self.FREQ_COLS]
            for col in self.ONEHOT_COLS:
                self.feature_names += [f'{col}_{v}' for v in self.onehot_maps[col]]
        return X

    def fit_transform(self, rows):
        self.fit(rows)
        X = self.transform(rows)
        n_scale = 2 + len(self.FREQ_COLS)
        means = X[:, :n_scale].mean(axis=0)
        stds  = X[:, :n_scale].std(axis=0) + 1e-8
        self.scale_params = {'mean': means, 'std': stds, 'n': n_scale}
        X[:, :n_scale] = (X[:, :n_scale] - means) / stds
        return X

    def transform_scaled(self, rows):
        X = self.transform(rows)
        n = self.scale_params['n']
        X[:, :n] = (X[:, :n] - self.scale_params['mean']) / self.scale_params['std']
        return X


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def log_loss(y_true, y_prob, eps=1e-7):
    """Binary cross-entropy — Kaggle's official CTR metric. Lower is better."""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def roc_auc(y_true, scores, n_thresholds=300):
    """ROC-AUC via threshold sweep. No sklearn."""
    thresholds = np.linspace(0, 1, n_thresholds)
    tprs, fprs = [], []
    pos = y_true.sum()
    neg = len(y_true) - pos
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / (pos + 1e-8))
        fprs.append(fp / (neg + 1e-8))
    tprs, fprs = np.array(tprs), np.array(fprs)
    order = np.argsort(fprs)
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    return trapezoid(tprs[order], fprs[order])


# ─────────────────────────────────────────────
# 5. NAIVE BAYES BASELINE
# ─────────────────────────────────────────────

class NaiveBayes:
    """
    Bernoulli Naive Bayes.

    Assumes each feature is independent given the class — the "naive" assumption.
    Works in log-space to avoid floating point underflow.
    Laplace smoothing (alpha) prevents zero probabilities for unseen values.

    Limitation: binarizes all features at 0, so frequency-encoded magnitudes
    are discarded. This is intentional — it sets a weak baseline for Logistic
    Regression to beat.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n, d   = X.shape
        n_pos  = int(y.sum())
        n_neg  = n - n_pos
        self.log_prior = np.array([np.log(n_neg / n), np.log(n_pos / n)])
        X_bin  = (X > 0).astype(np.float32)
        counts = np.zeros((d, 2))
        counts[:, 0] = X_bin[y == 0].sum(axis=0)
        counts[:, 1] = X_bin[y == 1].sum(axis=0)
        totals = np.array([n_neg, n_pos])
        self.log_like     = np.log((counts + self.alpha) / (totals + 2 * self.alpha))
        self.log_like_neg = np.log(1 - np.exp(self.log_like))
        print(f"  Fitted on {n:,} samples, {d} features")

    def predict_proba(self, X):
        X_bin  = (X > 0).astype(np.float32)
        ll1 = X_bin @ self.log_like[:, 1] + (1 - X_bin) @ self.log_like_neg[:, 1]
        ll0 = X_bin @ self.log_like[:, 0] + (1 - X_bin) @ self.log_like_neg[:, 0]
        log_sum = np.logaddexp(self.log_prior[0] + ll0, self.log_prior[1] + ll1)
        return np.exp(self.log_prior[1] + ll1 - log_sum)


# ─────────────────────────────────────────────
# 6. LOGISTIC REGRESSION
# ─────────────────────────────────────────────

class LogisticRegression:
    """
    Binary logistic regression with mini-batch SGD and L2 regularization.

    Forward pass:
      z = X @ w + b
      p = sigmoid(z) = 1 / (1 + exp(-z))

    Gradient update per mini-batch:
      dw = (X.T @ (p - y)) / batch_size  +  lambda_ * w
      db = mean(p - y)
      w -= lr * dw  |  b -= lr * db

    L2 regularization (lambda_):
      Penalizes large weights, reducing overfitting.
      Applied to w only — not the bias term.

    Learning rate decay:
      lr_t = lr / (1 + decay * epoch)
      Starts with large steps to move fast, shrinks to fine-tune near the minimum.
    """

    def __init__(self, lr=0.1, lambda_=0.001, batch_size=2048, n_epochs=20, decay=0.5):
        self.lr         = lr
        self.lambda_    = lambda_
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.decay      = decay

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        n, d     = X.shape
        self.w   = np.zeros(d, dtype=np.float32)
        self.b   = np.float32(0.0)
        rng      = np.random.default_rng(42)

        for epoch in range(self.n_epochs):
            idx    = rng.permutation(n)
            X_shuf = X[idx]
            y_shuf = y[idx]
            lr_t   = self.lr / (1.0 + self.decay * epoch)

            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n, self.batch_size):
                Xb  = X_shuf[start : start + self.batch_size]
                yb  = y_shuf[start : start + self.batch_size]
                p   = self._sigmoid(Xb @ self.w + self.b)
                err = p - yb
                self.w -= lr_t * ((Xb.T @ err) / len(yb) + self.lambda_ * self.w)
                self.b -= lr_t * float(err.mean())
                epoch_loss += log_loss(yb, p)
                n_batches  += 1

            if verbose and X_val is not None and (epoch + 1) % 5 == 0:
                vl = log_loss(y_val, self.predict_proba(X_val))
                print(f"    epoch {epoch+1:3d}  train={epoch_loss/n_batches:.4f}  val={vl:.4f}  lr={lr_t:.5f}")
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)


# ─────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val):
    lrs     = [0.5, 0.1, 0.01]
    lambdas = [0.0001, 0.001, 0.01]

    print(f"\n  {'lr':>6}  {'lambda':>8}  {'val_loss':>10}  {'val_auc':>9}")
    print("  " + "-" * 40)

    best_loss, best_params = float('inf'), {}
    for lr in lrs:
        for lam in lambdas:
            m     = LogisticRegression(lr=lr, lambda_=lam, n_epochs=20)
            m.fit(X_train, y_train, verbose=False)
            probs = m.predict_proba(X_val)
            vl    = log_loss(y_val, probs)
            va    = roc_auc(y_val, probs)
            mark  = " <-- best" if vl < best_loss else ""
            print(f"  {lr:>6.3f}  {lam:>8.4f}  {vl:>10.4f}  {va:>9.4f}{mark}")
            if vl < best_loss:
                best_loss, best_params = vl, {'lr': lr, 'lambda_': lam}

    print(f"\n  Best: lr={best_params['lr']}  lambda={best_params['lambda_']}  val_loss={best_loss:.4f}")
    return best_params


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'train')
    OUT_DIR   = os.path.dirname(__file__)

    # ── Step 1: Load ─────────────────────────
    print("\n[1/7] Reservoir sampling 200k rows...")
    rows = reservoir_sample(DATA_PATH, k=200_000)

    # ── Step 2: EDA ──────────────────────────
    print("\n[2/7] EDA...")
    run_eda(rows)

    # ── Step 3: Split ────────────────────────
    print("\n[3/7] Time-based split...")
    train_rows, val_rows, test_rows = time_based_split(rows)

    # ── Step 4: Features ─────────────────────
    print("\n[4/7] Feature engineering...")
    fe      = FeatureEngineer()
    X_train = fe.fit_transform(train_rows)
    X_val   = fe.transform_scaled(val_rows)
    X_test  = fe.transform_scaled(test_rows)
    y_train = np.array([int(r['click']) for r in train_rows], dtype=np.float32)
    y_val   = np.array([int(r['click']) for r in val_rows],   dtype=np.float32)
    y_test  = np.array([int(r['click']) for r in test_rows],  dtype=np.float32)
    print(f"  X_train : {X_train.shape}   X_val : {X_val.shape}   X_test : {X_test.shape}")
    print(f"  {len(fe.feature_names)} features total")

    # ── Step 5: Naive Bayes baseline ─────────
    print("\n[5/7] Naive Bayes baseline...")
    nb         = NaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    nb_probs   = nb.predict_proba(X_val)
    nb_logloss = log_loss(y_val, nb_probs)
    nb_auc     = roc_auc(y_val, nb_probs)
    print(f"  Val log-loss : {nb_logloss:.4f}")
    print(f"  Val ROC-AUC  : {nb_auc:.4f}")

    # ── Step 6: Logistic Regression ──────────
    print("\n[6/7] Logistic Regression — grid search on val...")
    best_params = grid_search(X_train, y_train, X_val, y_val)

    print(f"\n  Retraining best model on train + val combined...")
    X_tv        = np.vstack([X_train, X_val])
    y_tv        = np.concatenate([y_train, y_val])
    lr_model    = LogisticRegression(**best_params, n_epochs=20)
    lr_model.fit(X_tv, y_tv, verbose=False)
    lr_val_probs   = lr_model.predict_proba(X_val)
    lr_val_logloss = log_loss(y_val, lr_val_probs)
    lr_val_auc     = roc_auc(y_val, lr_val_probs)

    # ── Step 7: Final test evaluation ────────
    print("\n[7/7] Final evaluation on holdout test set (Oct 30)...")
    test_probs   = lr_model.predict_proba(X_test)
    test_logloss = log_loss(y_test, test_probs)
    test_auc     = roc_auc(y_test, test_probs)

    print(f"\n{'='*52}")
    print(f"{'MODEL':<28} {'LOG-LOSS':>10}  {'AUC':>8}")
    print(f"{'─'*52}")
    print(f"{'Naive Bayes (val)':<28} {nb_logloss:>10.4f}  {nb_auc:>8.4f}")
    print(f"{'Logistic Regression (val)':<28} {lr_val_logloss:>10.4f}  {lr_val_auc:>8.4f}")
    print(f"{'─'*52}")
    print(f"{'Logistic Regression (TEST)':<28} {test_logloss:>10.4f}  {test_auc:>8.4f}")
    print(f"{'='*52}")

    improvement = 100 * (nb_logloss - lr_val_logloss) / nb_logloss
    print(f"\n  Log-loss improvement over Naive Bayes: {improvement:.1f}%")

    # ── Save arrays ───────────────────────────
    for name, arr in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                      ('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
        np.save(os.path.join(OUT_DIR, f'{name}.npy'), arr)
    print(f"\n  Saved train/val/test arrays to {OUT_DIR}")
