# Presentation prep — decisions, reasoning, and anticipated questions

This file captures every design decision made in the project and the reasoning behind it, plus questions Jaspinder is likely to ask based on the Day 1–28 syllabus.

---

## Design decisions and the "why" behind each one

### Why reservoir sampling instead of loading all 40M rows?

Memory. 40M rows × 59 features × 4 bytes ≈ 8 GB just for the feature matrix. A typical laptop has 8–16 GB total. Even if it didn't crash, NumPy operations on 40M rows would be extremely slow.

The naive alternative — read the first 200k rows — is wrong because Avazu data is sorted chronologically. First 200k rows = Oct 21 midnight only. The model would never see afternoon, evening, weekends, or any later dates.

Reservoir sampling (Vitter's Algorithm R) streams through all 40M rows without loading them, keeps a 200k-row buffer where every row had equal probability of being selected. Result: ~20k rows per day across all 10 days.

### Why 200k rows specifically?

Practical balance. 200k fits comfortably in memory, trains in seconds, and gives ~20k rows per day (10 days × ~20k ≈ 200k). More rows would slow training without meaningfully improving the model at this scale. Fewer rows would under-represent rare feature values.

### Why time-based split instead of random?

Temporal leakage. A random split lets the model train on Oct 29 data while evaluating on Oct 22 data. In production, you always predict the future from the past. A model trained on future data appears better than it actually is — that gap only shows up when you deploy it.

Time-based split: train Oct 21–28, val Oct 29, test Oct 30. Mirrors production conditions.

### Why is Oct 30 the test set and never touched until the end?

Holdout discipline. Every time you look at test results and make a decision based on them (even choosing a threshold), you're leaking information from test into your model selection process. After enough such peeks, the test set effectively becomes part of training. Oct 30 is evaluated exactly once, at the end, with the model already fully locked.

### Why frequency encoding for high-cardinality columns?

site_id has ~1,800 unique values in the 200k sample. One-hot encoding creates 1,800 new columns — most nearly all zeros. That's slow to train, high-dimensional, and generalises poorly to IDs seen only once.

Frequency encoding replaces each ID with log(count + 1) from the training set. Popular sites get high values; rare sites get low values. One column. The log prevents a site with 100k impressions from dominating over one with 50k impressions.

Why log? Raw counts have a long tail. log(100001) − log(50001) ≈ 0.69, which is much more manageable than 50,000 apart in raw space.

### Why one-hot for low-cardinality columns?

device_type has 4 unique values: 0, 1, 2, 4. Keeping it as an integer implies device type 4 is "twice as much" as device type 2 — that's meaningless. One-hot creates one binary column per value, removing that false ordering. With only 4 values the columns stay compact (4 new cols vs 1,800).

### Why is leakage the core concern in feature engineering?

Frequency counts, one-hot vocabularies, and scaling parameters must all be computed on training rows only. If computed on all 200k rows before splitting, val and test rows would have contributed to their own encoding — the model would appear to perform better on val/test than it actually does on truly new data.

Concretely: if you compute z-score scaling on all 200k rows, the val and test means and stds "leak" into the scaler. The scaler knows too much about those rows. Fit on train only, transform val and test.

### Why Naive Bayes as a baseline?

A baseline exists to tell you whether your improvements are real. It should be easy to implement and deliberately weak. If your fancy model only beats an intentionally bad baseline by 1%, something is wrong with either the model or the data.

Bernoulli NB binarizes all features at 0 — a frequency-encoded site_id of 8.2 and 0.7 both become 1. This throws away the careful frequency encoding. That's intentional. It makes the baseline weak so there's room for better models to show improvement.

### Why Bernoulli NB and not Gaussian or Multinomial?

Bernoulli NB is designed for binary features. Since we binarize at 0, the features are binary — exactly what Bernoulli assumes. Gaussian NB would assume continuous features follow a normal distribution, which the binarized features don't. Multinomial NB is for count data. Bernoulli is the right fit for binary inputs.

### Why log-space for Naive Bayes?

NB multiplies many probabilities together. Each individual probability might be 0.3 or 0.7. Multiplying 59 of them: 0.5^59 ≈ 1.7 × 10^-18 — below float64's minimum representable positive value. The number underflows to 0, and you lose all signal. Working in log-space (summing log-probabilities instead of multiplying probabilities) keeps values in a safe numeric range. Laplace smoothing prevents log(0).

### Why mini-batch SGD for logistic regression?

Full-batch gradient descent: compute gradient on all 160k rows, take one step. Slow per step. Converges smoothly.

Stochastic GD: compute gradient on one row, take a step. Fast per step. Very noisy — zigzags toward the minimum.

Mini-batch: 2048 rows per step. Fast and stable — the best of both. 2048 is a common default; it fits in CPU cache and gives a good noise-smoothing balance.

### Why learning rate decay?

Large learning rate at the start → big steps → move toward the minimum quickly. But large steps near the minimum cause overshooting — the model bounces around. Decay shrinks the step over time: lr_t = lr / (1 + decay × epoch). Starts fast, fine-tunes at the end.

### Why L1 vs L2 regularization?

Both penalise large weights to prevent overfitting (fitting noise in training data). L2 shrinks all weights proportionally — no weight ever reaches exactly zero. L1 can push weights exactly to zero — automatic feature selection. In this project L1 and L2 tied (0.4111), suggesting most features are genuinely useful and L1 had nothing to zero out.

### Why grid search with 18 combinations instead of more?

Grid search is exhaustive — it trains a full model for every combination. 18 combinations (3 learning rates × 3 regularization strengths × 2 reg types) already takes ~1 minute. 100 combinations would take ~6 minutes for a marginal improvement. In practice, random search or Bayesian optimisation would be better for larger grids — but for 18 combinations, grid search is fine.

### Why Gini impurity for the decision tree?

Gini impurity measures how mixed a node is: 0 = pure (all one class), 0.5 = perfectly mixed. At each node, we try every feature and threshold, compute the weighted Gini impurity of the two child nodes, pick the split that minimises it most (maximises "information gain"). The alternative — entropy / information gain — gives nearly identical results in practice. Gini is slightly faster to compute (no log).

### Why max_depth=6 and min_samples_leaf=50?

Without constraints, a decision tree will grow until every leaf contains exactly one training row — it memorises the training data perfectly (100% train accuracy) but fails on new data (overfitting). max_depth=6 limits the tree to 6 levels of splits. min_samples_leaf=50 prevents splits on tiny subgroups. These are regularization for trees.

### Why only 20 sampled thresholds for tree splits?

A continuous feature can have thousands of unique values = thousands of possible thresholds. Testing all of them at every node on every tree is slow. We sample 20 random thresholds from the feature's range — fast enough to train in reasonable time while still finding good splits. A production tree would use percentiles or all unique values.

### Why Random Forest instead of a single decision tree?

A single decision tree has high variance — change the training data slightly and you get a completely different tree. Bootstrap sampling + feature subsampling forces 10 trees to be different from each other. Averaging their predictions smooths out the variance. This is the bias-variance tradeoff in action: each tree is slightly biased (due to seeing only √59 features per split), but the ensemble has much lower variance. Lower variance → better generalisation.

### Why 10 trees for Random Forest?

Diminishing returns. Going from 1 to 10 trees gives a big variance reduction. Going from 10 to 100 gives a smaller improvement but 10× the compute time. For a learning project on 160k rows, 10 trees gives a clear AUC improvement over a single tree without making training unbearably slow.

### Why √59 ≈ 7 features per split?

This is the standard rule of thumb for classification (for regression it's typically total_features / 3). The intuition: if you use all 59 features at every split, the best feature almost always wins and all trees look the same — highly correlated, no variance reduction from averaging. Using only 7 forces trees to explore different feature combinations, keeping them decorrelated.

### Why threshold tuning on val and not test?

Threshold tuning is a decision — "I'll call this a click if the predicted probability is above X." Any decision made by looking at test results moves you toward fitting test. The val set is specifically for these decisions. After tuning, apply the chosen threshold to test exactly once.

### Why does the default threshold of 0.5 fail here?

With 17% CTR, the model's predicted probabilities cluster around 0.15–0.25. Almost no prediction exceeds 0.5. Using 0.5 as the cutoff means the model almost never predicts click=1 — near-zero recall. The optimal threshold for this dataset is ~0.17–0.20, which is close to the actual CTR of 17% (intuitive: if 17% of impressions are clicks, a calibrated model's average prediction should be ~17%).

### Why is the Decision Tree the best model (selected for test)?

It won on val log-loss (0.407 vs RF's 0.411) and val AUC (0.671 vs RF's 0.677 — RF wins AUC but not log-loss). Log-loss is the primary metric because it penalises confident wrong predictions — directly relevant to ad bidding where you need calibrated probabilities, not just rankings. We pick the model with the best primary metric on val. Decision Tree.

### Why not ensemble the four models together?

Ensembling (averaging LR + DT + RF outputs) typically helps when models are diverse and each captures different signal. Here: RF already is an ensemble of DTs. Averaging RF + DT is mostly averaging DT with itself. LR + DT could add some diversity (linear vs non-linear), but the expected gain is small and the added complexity isn't worth it for a learning project. If this were a Kaggle submission, we'd ensemble.

### Why not PCA?

PCA reduces dimensionality by finding directions of maximum variance in a continuous feature space. Most of our 59 features are binary (one-hot columns). PCA on binary features destroys the interpretability of those columns and doesn't give a meaningful variance decomposition. Feature selection would be more appropriate (and L1 regularization already does implicit feature selection). PCA is powerful for image data, continuous sensor readings, etc. — not for this dataset.

### Why no kNN or SVM?

kNN computes the distance from every test point to every training point — O(n × m) for n test rows and m train rows. 160k training rows × 21k test rows = 3.4 billion distance computations. Infeasible in pure NumPy. SVM with a non-linear kernel has a similar scaling problem. Both are reasonable for small datasets (<10k rows) and become impractical here without library support.

### Why NumPy only?

Intentional constraint to demonstrate understanding of the underlying mechanics. Every formula — sigmoid, log-loss, Gini impurity, bootstrap sampling — is implemented explicitly. Using sklearn would hide all of that. The tradeoff is runtime: sklearn's compiled C code is 10–100× faster than our pure Python loops.

### Why a single Python file?

Simplicity and traceability. A single file can be read start to finish in one sitting. You can see the entire pipeline in one view: sampling → EDA → split → feature engineering → models → evaluation. Multi-file projects make sense when components need to be reused independently or when teams work in parallel. For a single-developer learning project, one file is clearer.

---

## Questions Jaspinder is likely to ask

### On sampling

**"How do you know reservoir sampling gives a truly uniform sample?"**
Vitter's Algorithm R has a mathematical proof: at every step, each of the rows seen so far has equal probability of being in the reservoir. By induction, when all 40M rows are processed, every row had probability 200k/40M of selection.

**"What if the data were clustered — would 200k rows be enough?"**
Good challenge. If Oct 21 had an unusual event (a viral ad, a sports event), our 20k rows from that day would over-represent it proportionally. Reservoir sampling handles the distribution of rows across time, but can't compensate for structural differences in the data itself. For this Avazu dataset the dates look similar, so 200k is fine.

**"Could you have used less than 200k rows?"**
Probably. The model results would be similar at 100k. The risk is rare feature values becoming even rarer — a site that appears 50 times in 200k might appear 25 times in 100k, which is still encodable. Below ~50k rows you'd start losing coverage of rare values.

### On the split

**"What if Oct 30 had unusual traffic? Is it a representative test?"**
Fair concern. A single day as the test set is vulnerable to day-specific effects. Ideally you'd test across multiple held-out periods. For this project, Oct 30 is what we have — and the small val-to-test gap (0.407 → 0.421 log-loss) suggests the model does generalise.

**"Why not k-fold cross-validation?"**
Standard k-fold shuffles the data before splitting — that introduces temporal leakage. Time-series cross-validation (train on earlier folds, validate on later ones) would be the right approach if we had more data. With only 10 days of data, a single temporal split is more practical.

### On feature engineering

**"Why log(count + 1) instead of just count?"**
The +1 handles zero counts (for values never seen in training that appear in val/test — they get log(0+1) = 0 rather than log(0) = -inf). The log compresses the scale — a site with 100k impressions and one with 50k impressions are meaningfully different, but not 2× different in terms of click signal.

**"What about features you didn't use? Like the C1-C21 anonymous columns?"**
All columns are included. C1–C21 are Avazu's anonymised categorical columns. We don't know what they represent, but we don't need to. If C18=1 correlates with lower CTR (which the LR weights confirm, w=−0.73), the model learns that whether or not we know C18 means "ad format" or "publisher category." The signal is real even if the label is hidden.

**"Would it help to create interaction features — like hour × device_type?"**
Probably yes. An interaction like "tablet user in the evening" might have different CTR than either tablet users or evening users alone. We didn't add interactions to keep the feature space interpretable and training fast. A production model would likely include them.

**"What's the difference between normalisation and standardisation?"**
Normalisation (min-max scaling): scales features to [0, 1]. Sensitive to outliers — one extreme value compresses everything else. Standardisation (z-score): mean 0, std 1. More robust to outliers. Z-score is generally preferred for gradient-based models like logistic regression.

### On models

**"Why did you choose log-loss as the primary metric and not AUC?"**
Log-loss penalises calibration — it cares not just about the ranking of predictions but about how confident the model is. For ad bidding, you're not just ranking ads, you're computing expected revenue = CTR × bid price. A mis-calibrated CTR estimate means you over- or under-bid. Log-loss is the right primary metric. AUC is a useful secondary metric for ranking ability.

**"What's the difference between PR-AUC and ROC-AUC?"**
ROC-AUC measures the area under the curve of true positive rate vs false positive rate across all thresholds. It's optimistic on imbalanced datasets because the false positive rate (FPR = FP / (FP + TN)) is small when negatives dominate — even a bad model looks okay. PR-AUC measures precision vs recall. Precision is sensitive to the number of false positives relative to true positives — more discriminating on imbalanced data. With 17% CTR, PR-AUC is more informative.

**"Why did the LR and RF have the same log-loss (0.411)?"**
Coincidence plus the limits of the dataset. Both models are likely hitting a ceiling — the Avazu features at 200k rows may not contain enough signal to push log-loss much below 0.41 with these methods. The difference lies in AUC and PR-AUC where RF has an edge, suggesting it ranks predictions better even if its probability calibration is similar.

**"What's the difference between L1 and L2 regularization intuitively?"**
L2 (ridge) adds a penalty proportional to the square of each weight: λ × w². The gradient is always non-zero, so weights shrink but never reach exactly zero. L1 (lasso) adds a penalty proportional to the absolute value: λ × |w|. The gradient is constant (±λ), which can push small weights all the way to zero — sparse solutions, implicit feature selection. In this project they tied, which means most features were genuinely contributing signal and L1 didn't zero anything useful out.

**"Why didn't you tune the Decision Tree hyperparameters with a grid search like you did for LR?"**
We did set max_depth=6 and min_samples_leaf=50, which are the two most important hyperparameters. A full grid search would have tried combinations of both. We didn't — partly for time, partly because the Decision Tree was selected based on the same val set that LR was tuned on, so they're on equal footing. In a production setting you'd grid search all models.

**"What's bootstrap sampling and why does it help?"**
Bootstrap sampling: draw n rows from the training set with replacement (so the same row can appear multiple times, and some rows never appear). Each bootstrap sample contains ~63% unique rows. By training 10 trees on 10 different bootstrap samples, each tree sees slightly different data → trees are diverse → averaging them reduces variance. This is bagging (bootstrap aggregating).

### On evaluation

**"Your precision is 0.271 on test — that seems low. Is the model actually useful?"**
Depends on the use case. Precision 0.271 means about 1 in 4 predicted clicks is a real click. Recall 0.578 means the model finds 57.8% of actual clicks. For ad ranking, you don't need to be right every time — you need to rank clickable ads higher than non-clickable ones. AUC 0.677 means the model correctly ranks a random click above a random non-click 67.7% of the time, compared to 50% for a random model. That's a 35% improvement over random, which translates to real revenue.

**"Why did you pick F1 to find the optimal threshold instead of something else?"**
F1 is the harmonic mean of precision and recall — it penalises extreme imbalance between the two. At threshold 0.5, recall is near zero — F1 forces you to find a threshold where both are reasonable. We could have optimised for a different metric (profit per impression if we knew the bid and revenue values). F1 is a good default when we don't have that business context.

**"What does the confusion matrix tell you?"**
At the optimal threshold: true positives (correctly predicted clicks) vs false positives (predicted click, actually no click) vs false negatives (predicted no click, actually clicked) vs true negatives. For a 17% CTR dataset the confusion matrix will always look imbalanced — most rows are true negatives. The interesting cells are false negatives (clicks you missed) and false positives (wasted impressions).

**"What's slice analysis and why does it matter?"**
Overall metrics hide subgroup failures. AUC 0.677 overall might mean AUC 0.72 on smartphones and AUC 0.57 on tablets. If 30% of your traffic is tablets, the overall number flatters the model. Slice analysis breaks performance by device_type and hour group to reveal where the model underperforms — usually where training data is thin or the feature signal is different.

### On what was skipped

**"Why no Pandas? It's the industry standard for data."**
Intentional constraint. This project demonstrates that you understand what Pandas does under the hood — dictionary operations, array math, column transformations — without relying on the abstraction. In practice you would use Pandas. The learning value was in doing it without.

**"Why no PCA?"**
PCA finds directions of maximum variance in continuous data. Most of our 59 features are binary (one-hot). PCA on binary columns gives decompositions that don't map back to interpretable features and don't help with click prediction. Feature selection (L1 regularization already handles this) would be more appropriate.

**"Why no SVM? It's a classic ML model."**
SVMs with non-linear kernels don't scale to 160k training rows without the kernel trick and specialised solvers — not implementable cleanly in NumPy. Linear SVM would be similar to logistic regression. Skipping it is defensible; the gap it would fill is already covered.

**"Your Random Forest has the highest AUC but you picked the Decision Tree as the best model — why?"**
We ranked by log-loss as the primary metric because we need calibrated probabilities (not just rankings) for ad bidding. Decision Tree: val log-loss 0.407. Random Forest: val log-loss 0.411. DT wins on the primary metric. RF wins on AUC and PR-AUC. The choice of primary metric is a design decision — and we made it explicitly.

### On runtime

**"How long does this take to run and does that matter?"**

About 5–6 minutes. Reservoir sampling takes 3–4 minutes (streaming 40M rows from disk). Grid search takes ~1 minute (18 model fits). Random Forest ~30 seconds.

Runtime matters differently at different stages:
- Training time: 5 minutes is fine for a daily retrain. Not fine if the model needs to retrain every minute.
- Inference time: this is usually the critical constraint. Serving a CTR prediction for a real ad impression needs to happen in <10ms. Our row-by-row Python tree traversal is far too slow for production — a compiled implementation would be 100× faster.
- The tradeoff in this project: we wrote readable, explicit code at the cost of speed. In production you'd use sklearn or LightGBM (compiled C/C++ under the hood).

### On broader ML concepts

**"What is overfitting vs underfitting?"**

Overfitting: the model memorises the training data including its noise. High training accuracy, low val/test accuracy. Signs: val loss increases while train loss decreases. Fix: regularization (L1/L2 for LR, max_depth for DT), more training data, simpler model.

Underfitting: the model is too simple to capture the underlying pattern. Both training and val accuracy are poor. Signs: train loss stays high. Fix: more complex model, better features, more data.

In this project: Decision Tree with max_depth=6 is intentionally constrained to avoid overfitting. The val-to-test gap (0.407 → 0.421) is small, suggesting we're neither severely over- nor underfitting.

**"What is bias-variance tradeoff?"**

Bias: error from wrong assumptions (underfitting). A linear model applied to a non-linear problem has high bias. Variance: error from sensitivity to training data (overfitting). A deep unconstrained decision tree has high variance — change a few rows and the tree looks completely different. Random Forest reduces variance by averaging many high-variance trees. L1/L2 regularization reduces variance in LR by penalising large weights.

**"What is data leakage?"**

Any situation where information from outside the training set reaches the model during training, making it appear to perform better than it would on truly new data. In this project: computing scaling parameters on all 200k rows (val/test rows influence the scaler), using a random instead of time-based split (training on future data), or choosing a threshold by looking at test results.

**"Why is imbalanced data a problem?"**

With 17% CTR, a model that always predicts "no click" achieves 83% accuracy. Accuracy is misleading. You need metrics that account for both classes: precision, recall, F1, AUC. Threshold tuning is often necessary because the default 0.5 assumes roughly balanced classes — not true here.

**"What would you do differently with more time?"**

- Tune Decision Tree hyperparameters with a grid search
- Try interaction features (hour × device_type, banner_pos × app_category)
- Ensemble LR + DT (adds real diversity — linear vs non-linear)
- Add confidence intervals to the val metrics (bootstrap the val set)
- Proper time-series cross-validation across multiple held-out periods
- Vectorise the tree's predict step for faster inference

---

## The core argument of this project

If Jaspinder asks "what's the most important thing you learned?", the answer is:

The model matters less than the data pipeline. Reservoir sampling gave us a representative sample. The time-based split gave us honest metrics. Leakage-free feature engineering gave the models good signal to work with. The Naive Bayes baseline (a deliberately weak model) already hit AUC 0.625 on clean, well-engineered features. Every improvement after that was incremental — 0.625 → 0.677.

The lesson from the quote — "a mediocre model with great features often beats a great model with mediocre features" — played out exactly in this project.
