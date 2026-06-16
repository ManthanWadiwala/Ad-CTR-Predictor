"""
Avazu CTR Prediction
====================
Pipeline:
  1.  Reservoir sample 200k rows from 40M  (memory-safe, spans all 10 days)
  2.  EDA — click rate, column distributions, CTR by hour
  3.  Time-based split — train Oct 21-28 / val Oct 29 / test Oct 30
  4.  Feature engineering — frequency encode, one-hot encode, z-score scale
  5.  Naive Bayes baseline         (from scratch, NumPy only)
  6.  Logistic Regression + L1/L2  (mini-batch SGD, learning curves, grid search)
  7.  Decision Tree                (Gini impurity, from scratch, NumPy only)
  8.  Random Forest                (bagging + feature subsampling, from scratch)
  9.  Threshold tuning             (find optimal threshold on val, not default 0.5)
  10. Slice analysis               (performance by device type and hour group)
  11. Final model comparison table
"""

import csv
import os
import numpy as np
from collections import Counter
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def reservoir_sample(filepath, k=200_000, seed=42):
    """
    Stream all rows and return k rows sampled uniformly at random.

    Taking the first 200k rows sequentially gives only Oct 21 — the model
    never sees later dates or peak hours. Reservoir sampling guarantees every
    row has an equal k/n chance of selection so the sample spans all 10 days.

    Algorithm:
      Fill reservoir with first k rows. For each subsequent row i, draw
      j in [0, i]. If j < k, replace reservoir[j] with the new row.
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
# EDA PLOTS
# ─────────────────────────────────────────────

def plot_eda(rows, out_dir):
    # CTR by hour of day
    hour_stats = {}
    for r in rows:
        h = int(r['hour']) % 100
        if h not in hour_stats:
            hour_stats[h] = [0, 0]
        hour_stats[h][0] += int(r['click'])
        hour_stats[h][1] += 1
    hours = sorted(hour_stats)
    ctrs  = [100 * hour_stats[h][0] / hour_stats[h][1] for h in hours]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(hours, ctrs, color='#378ADD', edgecolor='none')
    axes[0].axhline(y=16.98, color='#E24B4A', linestyle='--', linewidth=1, label='Overall CTR 16.98%')
    axes[0].set_xlabel('Hour of day')
    axes[0].set_ylabel('CTR (%)')
    axes[0].set_title('CTR by hour of day')
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(range(0, 24, 2))

    # Class imbalance
    n      = len(rows)
    clicks = sum(int(r['click']) for r in rows)
    axes[1].bar(['No click (83%)', 'Click (17%)'], [n - clicks, clicks],
                color=['#888780', '#1D9E75'], edgecolor='none')
    axes[1].set_ylabel('Row count')
    axes[1].set_title('Class imbalance')
    for i, v in enumerate([n - clicks, clicks]):
        axes[1].text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'eda_plots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved EDA plots → {path}")


def plot_roc_curves(y_true, probs_dict, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#888780', '#378ADD', '#1D9E75', '#BA7517']

    for ax_idx, (metric, ylabel) in enumerate([('roc', 'True positive rate'),
                                                ('pr',  'Precision')]):
        for (name, y_prob), color in zip(probs_dict.items(), colors):
            thresholds = np.linspace(0, 1, 300)
            xs, ys = [], []
            pos = y_true.sum()
            neg = len(y_true) - pos
            for t in thresholds:
                pred = (y_prob >= t).astype(int)
                tp = ((pred == 1) & (y_true == 1)).sum()
                fp = ((pred == 1) & (y_true == 0)).sum()
                fn = ((pred == 0) & (y_true == 1)).sum()
                if metric == 'roc':
                    xs.append(fp / (neg + 1e-8))
                    ys.append(tp / (pos + 1e-8))
                else:
                    prec = tp / (tp + fp + 1e-8)
                    rec  = tp / (tp + fn + 1e-8)
                    xs.append(rec)
                    ys.append(prec)
            order = np.argsort(xs)
            axes[ax_idx].plot(np.array(xs)[order], np.array(ys)[order],
                              label=name, color=color, linewidth=1.5)

        if metric == 'roc':
            axes[ax_idx].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
            axes[ax_idx].set_xlabel('False positive rate')
            axes[ax_idx].set_title('ROC curves')
        else:
            axes[ax_idx].set_xlabel('Recall')
            axes[ax_idx].set_title('Precision-Recall curves')

        axes[ax_idx].set_ylabel(ylabel)
        axes[ax_idx].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'roc_pr_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved ROC/PR curves → {path}")


def plot_model_comparison(results, out_dir):
    names     = list(results.keys())
    log_losses = [results[n]['log_loss'] for n in names]
    aucs       = [results[n]['auc'] for n in names]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, log_losses, w, label='Log-loss (lower is better)', color='#378ADD', edgecolor='none')
    ax.bar(x + w/2, aucs,       w, label='AUC (higher is better)',     color='#1D9E75', edgecolor='none')

    for i, (ll, auc) in enumerate(zip(log_losses, aucs)):
        ax.text(i - w/2, ll + 0.005, f'{ll:.3f}', ha='center', fontsize=9)
        ax.text(i + w/2, auc + 0.005, f'{auc:.3f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.35, 0.75)
    ax.set_title('Model comparison — validation set')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Saved model comparison → {path}")


# ─────────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────────

def time_based_split(rows):
    """
    Split by date — model always trains on the past, evaluates on the future.

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
      Replaces each value with log(count+1) from training set.

    One-hot encoding  — low-cardinality columns (device_type, banner_pos, etc.)
      One binary column per unique value. Removes false ordering.

    Z-score scaling  — frequency + temporal features only.
      Fit on train, applied to val/test — no leakage.
    """

    FREQ_COLS = [
        'site_id', 'site_domain', 'site_category',
        'app_id', 'app_domain', 'app_category',
        'device_id', 'device_ip', 'device_model',
        'C14', 'C17', 'C19', 'C20', 'C21'
    ]
    ONEHOT_COLS = ['C1', 'C18', 'device_type', 'device_conn_type', 'banner_pos', 'C15', 'C16']

    def __init__(self):
        self.freq_maps    = {}
        self.onehot_maps  = {}
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


def pr_auc(y_true, scores, n_thresholds=300):
    """
    Precision-Recall AUC — better than ROC-AUC for imbalanced classes.

    ROC-AUC can look good even when a model rarely predicts the minority class,
    because it rewards true negatives. PR-AUC only cares about how well the
    model finds positives — a harder, more honest metric for CTR data at 17% CTR.
    """
    thresholds  = np.linspace(0, 1, n_thresholds)
    precisions  = []
    recalls     = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        prec, rec, _ = precision_recall_f1(y_true, pred)
        precisions.append(prec)
        recalls.append(rec)
    precisions = np.array(precisions)
    recalls    = np.array(recalls)
    order      = np.argsort(recalls)
    trapezoid  = getattr(np, 'trapezoid', np.trapz)
    return float(trapezoid(precisions[order], recalls[order]))


def find_optimal_threshold(y_true, y_prob):
    """
    Sweep thresholds and return the one that maximises F1 on the val set.

    Why not use 0.5?
    With 17% positive class, predicting click=1 requires a probability > 0.5
    — but most CTR models output probabilities around 0.15-0.25. Using 0.5
    means the model almost never fires, collapsing recall to near zero.
    The optimal threshold is usually close to the actual CTR (~0.17 here).
    """
    thresholds = np.linspace(0.05, 0.60, 200)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        _, _, f1 = precision_recall_f1(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def confusion_matrix(y_true, y_pred):
    """Returns [[TN, FP], [FN, TP]]"""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    cm    = confusion_matrix(y_true, y_pred)
    tp    = cm[1, 1]
    fp    = cm[0, 1]
    fn    = cm[1, 0]
    prec  = tp / (tp + fp + 1e-8)
    rec   = tp / (tp + fn + 1e-8)
    f1    = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1


def print_eval(name, y_true, y_prob, threshold=0.5):
    """Print all metrics for a model in one call."""
    y_pred        = (y_prob >= threshold).astype(int)
    ll            = log_loss(y_true, y_prob)
    auc           = roc_auc(y_true, y_prob)
    prauc         = pr_auc(y_true, y_prob)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    cm            = confusion_matrix(y_true, y_pred)

    print(f"\n  {name}  (threshold={threshold:.2f})")
    print(f"  {'─'*44}")
    print(f"  Log-loss  : {ll:.4f}    ROC-AUC : {auc:.4f}    PR-AUC : {prauc:.4f}")
    print(f"  Precision : {prec:.4f}    Recall  : {rec:.4f}    F1     : {f1:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    return {'log_loss': ll, 'auc': auc, 'pr_auc': prauc,
            'precision': prec, 'recall': rec, 'f1': f1, 'threshold': threshold}


# ─────────────────────────────────────────────
# 5. NAIVE BAYES
# ─────────────────────────────────────────────

class NaiveBayes:
    """
    Bernoulli Naive Bayes.

    Assumes features are independent given the class. Works in log-space to
    avoid underflow. Laplace smoothing prevents zero probabilities for unseen
    values. Binarizes features at 0 — intentionally weak baseline.
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
        X_bin = (X > 0).astype(np.float32)
        ll1   = X_bin @ self.log_like[:, 1] + (1 - X_bin) @ self.log_like_neg[:, 1]
        ll0   = X_bin @ self.log_like[:, 0] + (1 - X_bin) @ self.log_like_neg[:, 0]
        log_sum = np.logaddexp(self.log_prior[0] + ll0, self.log_prior[1] + ll1)
        return np.exp(self.log_prior[1] + ll1 - log_sum)


# ─────────────────────────────────────────────
# 6. LOGISTIC REGRESSION
# ─────────────────────────────────────────────

class LogisticRegression:
    """
    Binary logistic regression with mini-batch SGD.

    Supports L1 and L2 regularization:
      L2 (Ridge): penalty = lambda_ * w         — shrinks all weights
      L1 (Lasso): penalty = lambda_ * sign(w)   — drives weak weights to zero

    Learning rate decay:
      lr_t = lr / (1 + decay * epoch)
    """

    def __init__(self, lr=0.1, lambda_=0.001, reg='l2', batch_size=2048, n_epochs=20, decay=0.5):
        self.lr         = lr
        self.lambda_    = lambda_
        self.reg        = reg
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
        history  = {'train_loss': [], 'val_loss': []}

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
                penalty = self.lambda_ * (np.sign(self.w) if self.reg == 'l1' else self.w)
                self.w -= lr_t * ((Xb.T @ err) / len(yb) + penalty)
                self.b -= lr_t * float(err.mean())
                epoch_loss += log_loss(yb, p)
                n_batches  += 1

            tl = epoch_loss / n_batches
            history['train_loss'].append(tl)
            if X_val is not None:
                vl = log_loss(y_val, self.predict_proba(X_val))
                history['val_loss'].append(vl)
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"    epoch {epoch+1:3d}  train={tl:.4f}  val={vl:.4f}  lr={lr_t:.5f}")
            elif verbose and (epoch + 1) % 5 == 0:
                print(f"    epoch {epoch+1:3d}  train={tl:.4f}  lr={lr_t:.5f}")

        self.history = history
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def print_learning_curve(self):
        print(f"\n  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}")
        print(f"  {'─'*32}")
        for i, tl in enumerate(self.history['train_loss']):
            vl_str = f"{self.history['val_loss'][i]:>10.4f}" if self.history['val_loss'] else ''
            print(f"  {i+1:>6}  {tl:>11.4f}  {vl_str}")

    def top_features(self, feature_names, n=10):
        """Return the n features with largest absolute weight."""
        idx = np.argsort(np.abs(self.w))[::-1][:n]
        return [(feature_names[i], float(self.w[i])) for i in idx]


# ─────────────────────────────────────────────
# 7. DECISION TREE
# ─────────────────────────────────────────────

class DecisionTree:
    """
    Binary decision tree for classification, built from scratch.

    Split criterion: Gini impurity
      Gini(node) = 1 - p² - (1-p)²
      where p = fraction of positive examples in the node.

    At each node, search all features × sampled thresholds for the split
    that maximises information gain = parent_gini - weighted_child_gini.

    Regularization:
      max_depth        — limits tree depth (controls overfitting)
      min_samples_leaf — minimum rows required in each leaf
    """

    class _Node:
        __slots__ = ['feature', 'threshold', 'left', 'right', 'value']
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature   = feature
            self.threshold = threshold
            self.left      = left
            self.right     = right
            self.value     = value  # leaf: P(click=1)

    def __init__(self, max_depth=6, min_samples_leaf=50, n_thresholds=20):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds     = n_thresholds
        self.root             = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)

    def _best_split(self, X, y):
        n            = len(y)
        parent_gini  = self._gini(y)
        best_gain    = -1.0
        best_feat    = None
        best_thresh  = None

        for feat in range(X.shape[1]):
            vals       = X[:, feat]
            thresholds = np.unique(
                np.percentile(vals, np.linspace(5, 95, self.n_thresholds))
            )
            for thresh in thresholds:
                left  = vals <= thresh
                nl    = int(left.sum())
                nr    = n - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gl   = self._gini(y[left])
                gr   = self._gini(y[~left])
                gain = parent_gini - (nl * gl + nr * gr) / n
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return self._Node(value=float(y.mean()))

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return self._Node(value=float(y.mean()))

        mask  = X[:, feat] <= thresh
        left  = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return self._Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        print(f"  Building tree (max_depth={self.max_depth}, "
              f"min_samples_leaf={self.min_samples_leaf})...")
        self.root = self._build(X, y.astype(np.float32), 0)
        return self

    def _predict_row(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict_proba(self, X):
        return np.array([self._predict_row(x, self.root) for x in X])


# ─────────────────────────────────────────────
# 8. RANDOM FOREST
# ─────────────────────────────────────────────

class RandomForest:
    """
    Ensemble of Decision Trees trained with bagging and feature subsampling.

    Why ensembles beat single trees:
      A single deep tree overfits — it memorises training data and has high
      variance (small data changes → very different tree). A shallow tree
      underfits — high bias, misses real patterns.

      Random Forest fixes this by averaging N trees that each make different
      errors. Two sources of randomness force the trees to be different:
        1. Bootstrap sampling — each tree trains on a random sample WITH
           replacement from the training data (~63% unique rows per tree).
        2. Feature subsampling — at each split, only sqrt(n_features) randomly
           chosen features are considered, not all 59. This decorrelates the
           trees so they don't all make the same mistakes.

      Averaging decorrelated trees reduces variance without increasing bias.
      This is the core bias-variance tradeoff insight from Day 17/22.
    """

    def __init__(self, n_trees=10, max_depth=6, min_samples_leaf=50,
                 n_thresholds=20, max_features='sqrt', seed=42):
        self.n_trees          = n_trees
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds     = n_thresholds
        self.max_features     = max_features
        self.seed             = seed
        self.trees            = []
        self.feature_subsets  = []

    def fit(self, X, y):
        n, d   = X.shape
        rng    = np.random.default_rng(self.seed)
        n_feat = int(np.sqrt(d)) if self.max_features == 'sqrt' else d
        self.trees           = []
        self.feature_subsets = []

        for i in range(self.n_trees):
            # Bootstrap sample — sample n rows WITH replacement
            boot_idx = rng.integers(0, n, size=n)
            X_boot   = X[boot_idx]
            y_boot   = y[boot_idx]

            # Random feature subset — pick n_feat features for this tree
            feat_idx = rng.choice(d, size=n_feat, replace=False)
            X_sub    = X_boot[:, feat_idx]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_thresholds=self.n_thresholds
            )
            print(f"  Tree {i+1}/{self.n_trees}  "
                  f"(bootstrap n={n:,}, features={n_feat}/{d})")
            tree.root = tree._build(X_sub, y_boot.astype(np.float32), 0)

            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)

        return self

    def predict_proba(self, X):
        # Average probability predictions across all trees
        preds = np.zeros(len(X), dtype=np.float64)
        for tree, feat_idx in zip(self.trees, self.feature_subsets):
            preds += tree.predict_proba(X[:, feat_idx])
        return preds / self.n_trees


# ─────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val):
    lrs     = [0.5, 0.1, 0.01]
    lambdas = [0.0001, 0.001, 0.01]
    regs    = ['l2', 'l1']

    print(f"\n  {'reg':>4}  {'lr':>6}  {'lambda':>8}  {'val_loss':>10}  {'val_auc':>9}")
    print("  " + "-" * 46)

    best_loss, best_params = float('inf'), {}
    for reg in regs:
        for lr in lrs:
            for lam in lambdas:
                m     = LogisticRegression(lr=lr, lambda_=lam, reg=reg, n_epochs=20)
                m.fit(X_train, y_train, verbose=False)
                probs = m.predict_proba(X_val)
                vl    = log_loss(y_val, probs)
                va    = roc_auc(y_val, probs)
                mark  = " <-- best" if vl < best_loss else ""
                print(f"  {reg:>4}  {lr:>6.3f}  {lam:>8.4f}  {vl:>10.4f}  {va:>9.4f}{mark}")
                if vl < best_loss:
                    best_loss   = vl
                    best_params = {'lr': lr, 'lambda_': lam, 'reg': reg}

    print(f"\n  Best: reg={best_params['reg']}  lr={best_params['lr']}"
          f"  lambda={best_params['lambda_']}  val_loss={best_loss:.4f}")
    return best_params


# ─────────────────────────────────────────────
# SLICE ANALYSIS
# ─────────────────────────────────────────────

def slice_analysis(rows, y_true, y_prob, label=''):
    """
    Break down model performance by device_type and hour group.
    Reveals where the model is strong or weak across subpopulations.
    """
    print(f"\n  Slice analysis — {label}")
    print(f"  {'Slice':<28}  {'N':>6}  {'CTR':>6}  {'AUC':>7}  {'LogLoss':>9}")
    print("  " + "-" * 62)

    # device_type slices
    device_types = sorted(set(r['device_type'] for r in rows))
    for dt in device_types:
        mask = np.array([r['device_type'] == dt for r in rows])
        if mask.sum() < 50:
            continue
        yt, yp = y_true[mask], y_prob[mask]
        auc = roc_auc(yt, yp) if yt.sum() > 0 else float('nan')
        ll  = log_loss(yt, yp)
        print(f"  {'device_type=' + dt:<28}  {mask.sum():>6}  "
              f"{yt.mean()*100:>5.1f}%  {auc:>7.4f}  {ll:>9.4f}")

    # hour group slices
    hour_groups = {
        'night  (00-05)': lambda h: h < 6,
        'morning(06-11)': lambda h: 6 <= h < 12,
        'afternoon(12-17)': lambda h: 12 <= h < 18,
        'evening(18-23)': lambda h: h >= 18,
    }
    hours = np.array([int(r['hour']) % 100 for r in rows])
    for name, fn in hour_groups.items():
        mask = np.array([fn(int(r['hour']) % 100) for r in rows])
        if mask.sum() < 50:
            continue
        yt, yp = y_true[mask], y_prob[mask]
        auc = roc_auc(yt, yp) if yt.sum() > 0 else float('nan')
        ll  = log_loss(yt, yp)
        print(f"  {name:<28}  {mask.sum():>6}  "
              f"{yt.mean()*100:>5.1f}%  {auc:>7.4f}  {ll:>9.4f}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'train')
    OUT_DIR   = os.path.dirname(__file__)

    results   = {}   # val metrics at default threshold
    results_t = {}   # val metrics at optimal threshold

    # ── Step 1: Load ─────────────────────────
    print("\n[1/9] Reservoir sampling 200k rows...")
    rows = reservoir_sample(DATA_PATH, k=200_000)

    # ── Step 2: EDA ──────────────────────────
    print("\n[2/9] EDA...")
    run_eda(rows)
    plot_eda(rows, OUT_DIR)

    # ── Step 3: Split ────────────────────────
    print("\n[3/9] Time-based split...")
    train_rows, val_rows, test_rows = time_based_split(rows)

    # ── Step 4: Features ─────────────────────
    print("\n[4/9] Feature engineering...")
    fe      = FeatureEngineer()
    X_train = fe.fit_transform(train_rows)
    X_val   = fe.transform_scaled(val_rows)
    X_test  = fe.transform_scaled(test_rows)
    y_train = np.array([int(r['click']) for r in train_rows], dtype=np.float32)
    y_val   = np.array([int(r['click']) for r in val_rows],   dtype=np.float32)
    y_test  = np.array([int(r['click']) for r in test_rows],  dtype=np.float32)
    print(f"  X_train:{X_train.shape}  X_val:{X_val.shape}  X_test:{X_test.shape}")
    print(f"  {len(fe.feature_names)} features total")

    # ── Step 5: Naive Bayes ──────────────────
    print("\n[5/10] Naive Bayes baseline...")
    nb       = NaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    nb_probs = nb.predict_proba(X_val)
    nb_thresh = find_optimal_threshold(y_val, nb_probs)
    results['Naive Bayes']   = print_eval("Naive Bayes (val)", y_val, nb_probs, threshold=0.5)
    results_t['Naive Bayes'] = print_eval("Naive Bayes (val, optimal threshold)",
                                           y_val, nb_probs, threshold=nb_thresh)

    # ── Step 6: Logistic Regression ──────────
    print("\n[6/10] Logistic Regression — grid search (L1 and L2) on val...")
    best_params = grid_search(X_train, y_train, X_val, y_val)

    print(f"\n  Retraining best config with learning curves...")
    best_lr = LogisticRegression(**best_params, n_epochs=20)
    best_lr.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    best_lr.print_learning_curve()

    lr_val_probs = best_lr.predict_proba(X_val)
    lr_thresh    = find_optimal_threshold(y_val, lr_val_probs)
    results['Logistic Regression']   = print_eval(
        f"Logistic Regression [{best_params['reg'].upper()}] (val)",
        y_val, lr_val_probs, threshold=0.5)
    results_t['Logistic Regression'] = print_eval(
        f"Logistic Regression [{best_params['reg'].upper()}] (val, optimal threshold)",
        y_val, lr_val_probs, threshold=lr_thresh)

    print(f"\n  Top 10 features by weight magnitude:")
    for feat, weight in best_lr.top_features(fe.feature_names, n=10):
        direction = "→ click" if weight > 0 else "→ no click"
        print(f"    {feat:<30}  w={weight:+.4f}  {direction}")

    # ── Step 7: Decision Tree ────────────────
    print("\n[7/11] Decision Tree (from scratch, Gini impurity)...")
    dt = DecisionTree(max_depth=6, min_samples_leaf=50)
    dt.fit(X_train, y_train)
    dt_val_probs = dt.predict_proba(X_val)
    dt_thresh    = find_optimal_threshold(y_val, dt_val_probs)
    results['Decision Tree']   = print_eval("Decision Tree (val)",
                                             y_val, dt_val_probs, threshold=0.5)
    results_t['Decision Tree'] = print_eval("Decision Tree (val, optimal threshold)",
                                             y_val, dt_val_probs, threshold=dt_thresh)

    # ── Step 8: Random Forest ─────────────────
    print("\n[8/11] Random Forest (10 trees, bootstrap + feature subsampling)...")
    rf = RandomForest(n_trees=10, max_depth=6, min_samples_leaf=50)
    rf.fit(X_train, y_train)
    rf_val_probs = rf.predict_proba(X_val)
    rf_thresh    = find_optimal_threshold(y_val, rf_val_probs)
    results['Random Forest']   = print_eval("Random Forest (val)",
                                             y_val, rf_val_probs, threshold=0.5)
    results_t['Random Forest'] = print_eval("Random Forest (val, optimal threshold)",
                                             y_val, rf_val_probs, threshold=rf_thresh)

    # ── Plots ────────────────────────────────
    print("\nGenerating plots...")
    probs_dict = {
        'Naive Bayes':        nb_probs,
        'Logistic Regression': lr_val_probs,
        'Decision Tree':      dt_val_probs,
        'Random Forest':      rf_val_probs,
    }
    plot_roc_curves(y_val, probs_dict, OUT_DIR)
    plot_model_comparison(results, OUT_DIR)

    # ── Step 9: Threshold summary ────────────
    print("\n[9/11] Optimal threshold summary...")
    print(f"\n  {'Model':<28}  {'Default t=0.5 F1':>17}  {'Optimal t':>10}  {'Optimal F1':>11}")
    print("  " + "─" * 72)
    for name in results:
        f1_default = results[name]['f1']
        f1_opt     = results_t[name]['f1']
        t_opt      = results_t[name]['threshold']
        print(f"  {name:<28}  {f1_default:>17.4f}  {t_opt:>10.3f}  {f1_opt:>11.4f}")

    # ── Step 9: Slice analysis ───────────────
    print("\n[10/11] Slice analysis on val set...")
    slice_analysis(val_rows, y_val, lr_val_probs, label='Logistic Regression')
    slice_analysis(val_rows, y_val, dt_val_probs, label='Decision Tree')
    slice_analysis(val_rows, y_val, rf_val_probs, label='Random Forest')

    # ── Step 11: Final test evaluation ────────
    print("\n[11/11] Final evaluation on holdout test set (Oct 30)...")
    # Pick best model on val log-loss
    best_name = min(results, key=lambda k: results[k]['log_loss'])
    best_probs_map = {
        'Naive Bayes': nb_probs,
        'Logistic Regression': lr_val_probs,
        'Decision Tree': dt_val_probs,
        'Random Forest': rf_val_probs,
    }
    best_thresh_map = {
        'Naive Bayes': nb_thresh,
        'Logistic Regression': lr_thresh,
        'Decision Tree': dt_thresh,
        'Random Forest': rf_thresh,
    }
    best_test_map = {
        'Naive Bayes': nb.predict_proba(X_test),
        'Logistic Regression': best_lr.predict_proba(X_test),
        'Decision Tree': dt.predict_proba(X_test),
        'Random Forest': rf.predict_proba(X_test),
    }
    print(f"\n  Best model on val: {best_name}")
    test_probs  = best_test_map[best_name]
    best_t      = best_thresh_map[best_name]
    test_pred   = (test_probs >= best_t).astype(int)
    test_ll     = log_loss(y_test, test_probs)
    test_auc_v  = roc_auc(y_test, test_probs)
    test_prauc  = pr_auc(y_test, test_probs)
    test_prec, test_rec, test_f1 = precision_recall_f1(y_test, test_pred)

    print(f"\n{'='*72}")
    print(f"{'MODEL':<28} {'LOGLOSS':>8} {'ROC-AUC':>8} {'PR-AUC':>7} "
          f"{'PREC':>6} {'REC':>6} {'F1':>6}")
    print(f"{'─'*72}")
    print("  — default threshold (0.5) —")
    for name, m in results.items():
        print(f"  {name:<26} {m['log_loss']:>8.4f} {m['auc']:>8.4f} {m['pr_auc']:>7.4f} "
              f"{m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f}")
    print("  — optimal threshold (tuned on val) —")
    for name, m in results_t.items():
        print(f"  {name:<26} {'':>8} {'':>8} {'':>7} "
              f"{m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f}  t={m['threshold']:.3f}")
    print(f"{'─'*72}")
    print(f"  {f'Best ({best_name}) — TEST':<26} {test_ll:>8.4f} {test_auc_v:>8.4f} "
          f"{test_prauc:>7.4f} {test_prec:>6.4f} {test_rec:>6.4f} {test_f1:>6.4f}"
          f"  t={best_t:.3f}")
    print(f"{'='*72}")

    nb_ll = results['Naive Bayes']['log_loss']
    lr_ll = results['Logistic Regression']['log_loss']
    dt_ll = results['Decision Tree']['log_loss']
    rf_ll = results['Random Forest']['log_loss']
    print(f"\n  LR vs NB : {100*(nb_ll-lr_ll)/nb_ll:.1f}% log-loss reduction")
    print(f"  DT vs NB : {100*(nb_ll-dt_ll)/nb_ll:.1f}% log-loss reduction")
    print(f"  RF vs NB : {100*(nb_ll-rf_ll)/nb_ll:.1f}% log-loss reduction")

    # ── Save arrays ───────────────────────────
    for name, arr in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                      ('y_train', y_train), ('y_val', y_val),  ('y_test', y_test)]:
        np.save(os.path.join(OUT_DIR, f'{name}.npy'), arr)
    print(f"\n  Saved train/val/test arrays to {OUT_DIR}")
