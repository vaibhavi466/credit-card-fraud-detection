# Learning Notes — Credit Card Fraud Detection

> Personal study notes. Written to be re-readable months later when I've forgotten the details. Not for recruiters — this is for me. Think of it as a mini-textbook for the concepts this project uses.

---

## 1. Why SMOTE works — the actual mechanism

### The problem: naive oversampling
If you duplicate fraud transactions, you repeat the exact same feature vectors in training. The model memorises those specific points — it learns "flag exactly this set of values" — and overfits badly. Generalisation suffers.

### What SMOTE does instead
SMOTE (Synthetic Minority Over-sampling TEchnique, Chawla et al. 2002) creates **new, synthetic** fraud examples by interpolating in feature space:

```
Given minority sample x:
1. Find the k=5 nearest minority-class neighbours of x
2. Pick one neighbour x_n at random
3. Draw random weight λ ∈ [0, 1]
4. New synthetic sample = x + λ × (x_n - x)
```

In plain English: pick a real fraud, pick another real fraud nearby, and generate a point somewhere between them on the straight line connecting them. That new point is a "could have existed" fraud transaction.

### Why this generalises better
The synthetic sample is *plausible* — it lives in the minority-class neighbourhood — but it's *novel* — it wasn't seen in training. This forces the model to learn the shape of the fraud region, not memorise specific points.

### The critical caveat: apply only to training data
If you SMOTE before splitting:
- Synthetic sample S is generated from real frauds F1 and F2
- If F1 ends up in the test set after splitting, S (which was partly generated from F1) will be in training
- The model learns patterns from F1 indirectly through S, then is "tested" on F1 — this is leakage
- The correct order: split → SMOTE on training fold only

### SMOTE+Tomek: the bonus cleaning step
After SMOTE, some synthetic samples land near the boundary between classes. Tomek link removal identifies pairs (one legit, one fraud) that are each other's nearest neighbours — these are the most ambiguous, borderline examples. Removing them gives the classifier a cleaner decision boundary to learn.

---

## 2. Why AUPRC beats ROC-AUC under heavy imbalance

### ROC curve: TPR vs FPR
```
TPR = TP / (TP + FN)  = "what fraction of real frauds did we catch?"
FPR = FP / (FP + TN)  = "what fraction of real legits did we wrongly flag?"
```

The denominator of FPR is FP + TN. In our dataset: TN ≈ 227,000. Even if we flag 1,000 legitimate transactions falsely, FPR = 1000 / 228000 ≈ 0.004 — barely visible on the ROC curve. A terrible classifier can have ROC-AUC > 0.95 on this data.

### PR curve: Precision vs Recall
```
Precision = TP / (TP + FP)  = "of the ones we flagged, what % were real fraud?"
Recall    = TP / (TP + FN)  = "of all real frauds, what % did we catch?"
```

Precision directly penalises false positives — every false alarm makes it worse. The random classifier baseline for AUPRC is the class prevalence (0.172%), not 0.5. So AUPRC ≥ 0.05 already represents a 29× lift over random.

### Intuition
ROC-AUC measures "can the model rank a random fraud above a random legit?" AUPRC measures "given everything it flagged as fraud, how useful is that list?" For operational fraud teams who review flagged transactions, AUPRC directly measures the quality of what lands in their queue.

---

## 3. What a SHAP value actually represents

### The motivation: feature importance is global, but explanations need to be local
Standard feature importance (e.g., Gini importance in Random Forest) tells you "V14 is important on average across all predictions." But for a specific transaction that got flagged, you want to know: "which features caused *this particular* prediction to be 0.87?"

### Game theory foundation (Shapley values)
Imagine each feature as a "player" in a cooperative game, and the model's output as the "payout" to be divided. The Shapley value for player (feature) i is its *fair share* of the payout — computed as its average marginal contribution across all possible orderings of features joining the coalition.

For feature i's Shapley value:
- Consider all subsets S of features not containing i
- For each subset, compute the difference in model output with vs. without feature i
- Average this over all subsets (weighted by subset size)

This gives a unique, theoretically justified allocation of the prediction to features.

### Practical meaning for XGBoost (TreeSHAP)
For a single fraud transaction with fraud probability 0.87:
- Base value (expected model output): ~0.003 (the fraud rate)
- SHAP for V14: +0.4 → "V14 pushed the probability up by 0.4"
- SHAP for V4: +0.3 → "V4 pushed it up by 0.3"
- SHAP for Amount: -0.05 → "Amount (low in this case) pushed it slightly toward legit"
- Sum of all SHAPs + base value = 0.87

The waterfall plot visualises this cascade. TreeSHAP (Lundberg et al. 2018) computes exact Shapley values for tree models in polynomial time, making it practical.

### Why this matters for trust
"Why did the model flag this transaction?" has a concrete answer: because V14 had an unusually negative value and V4 was in an unusual range, together contributing +0.7 to the fraud score. A fraud analyst can look at that explanation and decide whether to investigate or release the hold.

---

## 4. The leakage guard — what exactly breaks without it

### Scenario 1: Scaling the full dataset before splitting

```python
# WRONG — leaky
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)        # uses ALL data including test
X_train, X_test = train_test_split(X_scaled, ...)

# RIGHT
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # only training statistics
X_test = scaler.transform(X_test)         # apply train statistics to test
```

**What breaks:** The scaler's mean/std incorporate test-set statistics. The scaled test features are "centred" on the true test mean, not the train mean. The model sees slightly different feature scales during training vs. what it would see in production (where you'd only have train statistics available). The effect is usually small (1-2% metric inflation) but it's philosophically wrong and easy to avoid.

### Scenario 2: SMOTE before splitting (more serious)

```python
# WRONG — leaky
X_res, y_res = SMOTE().fit_resample(X, y)   # includes test data
X_train, X_test = train_test_split(X_res, y_res, ...)

# RIGHT
X_train, X_test = train_test_split(X, y, ...)
X_res, y_res = SMOTE().fit_resample(X_train, y_train)  # only training fold
```

**What breaks:** SMOTE generates synthetic samples by interpolating between real fraud samples. If a real fraud transaction ends up in the test set, synthetic training samples were partly derived from it. The model learns patterns from test-adjacent data and achieves artificially high recall. Depending on the split, this can inflate AUPRC by 5-15 points — the difference between a paper result and a production result.

**Interview answer:** "Resampling after splitting is trivial to implement but catches a mistake that's extremely common in published code. I implemented it as a design constraint — the resampling functions in `src/resampling.py` only accept `X_train, y_train` by signature, so it's architecturally impossible to accidentally pass test data."

---

## 5. Cost-sensitive learning — the business framing

### The decision problem
Every transaction, the model has to decide: flag it or let it through? The right answer depends on:
- How bad is it if we miss a fraud? → proportional to the fraud amount
- How bad is it if we flag a legit? → proportional to customer friction cost

This is a threshold selection problem, not a model problem. The model outputs probabilities; the business sets the threshold based on its cost structure.

### The maths
At threshold t:
- FN(t) = number of frauds we miss at threshold t
- FP(t) = number of legits we flag at threshold t
- Total cost(t) = FN(t) × fn_cost + FP(t) × fp_cost

We want min_t[Total cost(t)]. This is just a 1D minimisation over a sweep of t values.

### Why the optimal threshold changes with cost assumptions
If fn_cost >> fp_cost (high fraud amounts, low friction cost):
→ Optimal threshold is low → we catch more fraud even at the cost of many false alarms

If fp_cost >> fn_cost (expensive customer friction, small fraud amounts):
→ Optimal threshold is high → we tolerate more misses to protect legit customers

The model doesn't change. The threshold changes. This is why threshold selection is a business decision, not just an ML decision.

### My implementation
I compute fn_cost as the mean fraud Amount from the actual dataset (at runtime — not hardcoded). FP cost = $5 is illustrative and clearly labeled as such. The cost curve is saved to `reports/figures/cost_curve.png` so a stakeholder can see the trade-off visually.

---

## 6. The autoencoder as an anomaly detector

### Architecture
An autoencoder is a neural network that learns to compress then reconstruct its input:
```
Input (29 features) → Encoder → Bottleneck (8 features) → Decoder → Reconstructed input (29 features)
```

Trained with a reconstruction loss (MSE): the model learns to represent the *most important* information in 8 dimensions and discard the rest.

### The fraud detection trick
Train ONLY on legitimate transactions. The autoencoder becomes an expert at reconstructing "normal" transaction patterns. When it sees a fraud transaction:
- The fraud pattern is outside the manifold of normal transactions
- The reconstruction is poor — high MSE
- High reconstruction error → anomaly flag

### When this beats supervised learning
1. **Zero labels:** No confirmed fraud history (new product, new market)
2. **Concept drift:** Fraudsters develop new patterns the supervised model never saw
3. **Zero-day fraud:** Entirely novel fraud type → supervised model has no training signal for it; autoencoder flags it because it deviates from normal

### Limitations vs. supervised
- If fraud patterns overlap significantly with legit patterns in feature space, reconstruction error won't separate them well
- Need a threshold on reconstruction error — this is a hyperparameter requiring a labeled validation set (which partially defeats the purpose)
- Generally lower AUPRC than supervised models when good labels exist

### How I'd use both together
Tier 1: Autoencoder flags anything with high reconstruction error → broad net
Tier 2: Supervised model scores the flagged subset → precision filter
Result: Higher recall than pure supervised (catches novel fraud) with manageable FP rate (supervised model filters the autoencoder's false alarms)
