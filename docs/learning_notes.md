# Learning Notes — Credit Card Fraud Detection

> Personal study notes. Written to be re-readable months later when I've forgotten the details. Not for recruiters — this is for me. Think of it as a technical reference guide for the ML concepts this project implements.

---

## 1. Why SMOTE Works — The Actual Mechanism

### The Problem: Naive Oversampling
If you duplicate fraud transactions, you repeat exact feature vectors in training. The model memorises those specific data points — learning to "flag exactly this set of values" — and overfits badly. Generalisation on unseen transactions suffers.

### What SMOTE Does Instead
SMOTE (Synthetic Minority Over-sampling TEchnique, Chawla et al. 2002) creates **new, synthetic** fraud examples by interpolating in feature space:

```
Given minority sample x:
1. Find the k=5 nearest minority-class neighbours of x
2. Pick one neighbour x_n at random
3. Draw random weight λ ∈ [0, 1]
4. New synthetic sample = x + λ × (x_n - x)
```

In plain English: pick a real fraud, pick another real fraud nearby, and generate a point somewhere on the straight line connecting them. That new point represents a plausible synthetic fraud transaction.

### Why This Generalises Better
The synthetic sample is *plausible* — it lives in the minority-class neighbourhood — but *novel* — it wasn't seen in raw data. This forces the classifier to learn the boundary shape of the fraud region rather than memorising individual training instances.

### The Critical Caveat: Apply Only to Training Data
If you apply SMOTE before splitting:
- Synthetic sample $S$ is generated from real frauds $F_1$ and $F_2$.
- If $F_1$ ends up in the test set after splitting, $S$ (which was partly generated from $F_1$) will be in training.
- The model learns patterns from $F_1$ indirectly through $S$, then is tested on $F_1$ — an explicit data leakage path.
- The correct order: **Split $\rightarrow$ Apply SMOTE only on training fold.**

### SMOTE+Tomek: Boundary Cleaning
After SMOTE, some synthetic samples land near the boundary between classes. Tomek link removal identifies pairs (one legitimate, one fraud) that are each other's nearest neighbours — the most ambiguous, borderline examples. Removing them cleans the decision boundary.

---

## 2. Why AUPRC Beats ROC-AUC Under Heavy Imbalance

### ROC Curve: TPR vs. FPR
$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad (\text{Recall})$$
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

In our dataset, legitimate transactions total ~284,315 ($\text{TN} \approx 56,843$ in test). Even if we flag 500 legitimate transactions falsely, $\text{FPR} = 500 / 56,843 \approx 0.0088$ — barely moving the point on the ROC curve. A weak classifier can achieve $\text{ROC-AUC} > 0.95$ while delivering dismal precision on the minority class.

### PR Curve: Precision vs. Recall
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Precision directly penalises false positives — every false alarm lowers precision. The random-classifier baseline for AUPRC equals the fraud prevalence ($0.172\% = 0.00172$), not $0.50$. Thus, an AUPRC of $0.8771$ represents a massive, mathematically defensible improvement over random guessing.

### Operational Relevance
For fraud operations teams reviewing flagged alerts, Precision measures queue purity (what fraction of alerts are actual fraud), while Recall measures coverage (what fraction of total fraud was stopped). AUPRC directly evaluates this trade-off.

---

## 3. What SHAP Values Represent

### Local Feature Attribution via Game Theory
Standard feature importance (e.g. Gini importance) indicates average importance across the dataset. For a specific flagged transaction, risk analysts need to know: *"Which features caused this specific transaction to receive a high fraud risk score?"*

Shapley values (from cooperative game theory) treat features as players in a game and the prediction as the payoff. The Shapley value for feature $i$ is its fair share of the payout — computed as its average marginal contribution across all feature subsets.

### TreeSHAP Implementation
For a tree ensemble (Random Forest or XGBoost), TreeSHAP (Lundberg et al. 2018) calculates exact Shapley values in polynomial time:
- Base value (expected model output): average prediction across background samples.
- Feature SHAP values: push the score above or below the base value.
- Sum of all feature SHAP values $+$ base value $=$ model prediction score.

In our dataset, TreeSHAP identifies components `V4`, `V14`, and `V12` as the dominant feature drivers for fraud classification.

---

## 4. Leakage Guards & Methodological Correctness

### 3-Way Stratified Partitioning (60% Train / 20% Val / 20% Test)
To ensure test evaluation is 100% out-of-sample and unbiased:
1. Full dataset is partitioned into 80% Train-Val and 20% Test.
2. Train-Val is partitioned into 75% Train (60% of total) and 25% Validation (20% of total).
3. The 20% Test set is held out completely untouched until final scoring.

### Scaler Fit Isolation
```python
# RIGHT — leakage-free
X_train, X_val, X_test = stratified_train_val_test_split(df)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['Amount', 'Time']])  # fit on Train only
X_val_scaled = scaler.transform(X_val[['Amount', 'Time']])        # transform using Train stats
X_test_scaled = scaler.transform(X_test[['Amount', 'Time']])      # transform using Train stats
```

Fitting the scaler on the full dataset leaks test set distribution statistics (means and standard deviations) into training features.

### Train-Only FN Cost Derivation
Calculating the False Negative (missed fraud) cost parameter from full dataset labels introduces subtle label leakage. In our hardened pipeline, the mean fraud amount ($130.15) is derived strictly from `X_train` row indices.

---

## 5. Cost-Sensitive Threshold Selection

### The Decision Optimization Problem
Machine Learning models output continuous scores or probabilities. The decision threshold converts scores into discrete binary actions (decline vs. approve). The optimal threshold minimizes expected monetary loss:

$$\text{Total Cost}(t) = \text{FN}(t) \times \text{Cost}_{\text{FN}} + \text{FP}(t) \times \text{Cost}_{\text{FP}}$$

- $\text{Cost}_{\text{FN}} = \$130.15$ (Mean fraud transaction amount derived strictly from training data).
- $\text{Cost}_{\text{FP}} = \$5.00$ (Illustrative customer friction cost assumption).

### Validation Locking
Threshold sweeps are performed exclusively on validation predictions (`X_val`). The threshold that minimizes total validation cost ($0.0600$) is locked and applied to the untouched test set. The test set is never used to select or adjust the threshold.

---

## 6. Unsupervised Autoencoder Anomaly Detection

### Architecture & Objective
An autoencoder (`MLPRegressor`) compresses inputs (30 features) into a low-dimensional bottleneck (8 nodes) and reconstructs them:
```
Input (30 features) -> Encoder -> Bottleneck (8 features) -> Decoder -> Reconstruction (30 features)
```

### Training Protocol
1. Train **strictly** on legitimate transactions from `X_train` (170,588 samples).
2. The autoencoder learns to reconstruct normal transaction patterns with minimal Mean Squared Error (MSE).
3. Reconstruction error on `X_val` legitimate samples defines the decision threshold (95th percentile = 0.5168).
4. Evaluate once on `X_test`. Transactions exceeding threshold 0.5168 are flagged as anomalies.

### Strategic Role
The autoencoder acts as an unsupervised zero-day anomaly detector to catch novel fraud patterns that supervised models may miss due to lack of historical training signals.
