# Model Card — Credit Card Fraud Detection

**Framework:** scikit-learn (Random Forest, imbalanced-learn, SHAP)

---

## Model Details

| Property | Value |
|---|---|
| Model type | Random Forest |
| Imbalance strategy | SMOTE (train-fold only) |
| Primary metric | AUPRC |
| Training data | ULB creditcard dataset (Sep 2013) |
| Input features | V1–V28 (PCA), Amount (scaled), Time (scaled) |
| Output | Fraud probability ∈ [0, 1] |
| Decision threshold | Cost-optimal (see reports/metrics.json) |

---

## Intended Use

**Intended users:** Fraud analysts, risk teams at financial institutions.

**Intended use cases:**
- Real-time or batch scoring of credit card transactions to prioritise analyst review
- Portfolio demonstration of ML pipeline engineering practices
- Educational reference for imbalanced classification

**Out-of-scope uses:**
- **Do not use** as the sole basis for irreversible actions against customers (account closure, criminal reporting) without human review
- **Do not use** on non-credit-card transaction data without retraining and validation
- **Do not use** as a fairness-certified model — see Fairness section below

---

## Performance

| Metric | Value (at default threshold 0.50) | Value (at cost-optimal threshold 0.32) |
|---|---|---|
| AUPRC | 0.8747 | 0.8747 |
| ROC-AUC | 0.9731 | 0.9731 |
| Precision (fraud class) | 0.8454 | 0.7586 |
| Recall (fraud class) | 0.8367 | 0.8980 |
| F1 (fraud class) | 0.8410 | 0.8224 |

**Baseline comparison:** Logistic Regression with `class_weight='balanced'` scores AUPRC ≈ 0.7159. The Random Forest + SMOTE pipeline improves this by ≈ 0.1588 AUPRC points.

---

## Training Data

- **Dataset:** ULB Credit Card Fraud Detection (Kaggle: mlg-ulb/creditcardfraud)
- **Period:** September 2013, European cardholders
- **Size:** 284,807 transactions; 492 fraud (0.172%)
- **Features:** PCA-anonymised (V1–V28); raw Amount and Time
- **Split:** 80% train / 20% test (stratified by Class, random_state=42)
- **Resampling:** SMOTE applied to training fold only — never touches test set

---

## Limitations

1. **Temporal:** The model was trained on 2013 data. Fraud patterns change significantly over years. Performance on contemporary data is unknown and likely degraded.

2. **Geographic:** Data reflects European cardholder behaviour and fraud patterns. Applicability to other regions is unvalidated.

3. **PCA opacity:** V1–V28 are redacted PCA components. It is not possible to determine whether these features encode any protected attributes (age, location, race, etc.) — they are combinations of the original features which are confidential.

4. **Static threshold:** The cost-optimal threshold was computed using a $5 illustrative FP cost. A different business assumption changes the threshold significantly.

5. **No temporal validation:** The 80/20 split is random, not time-based. In practice, a production model would be evaluated on a held-out future period to avoid look-ahead bias.

---

## Fairness

> [!WARNING]
> **False Positives block legitimate customers.** A False Positive means a real customer's legitimate transaction is declined. This can disproportionately impact customers who make unusual-but-legitimate transactions (e.g., large purchases abroad). At the cost-optimal threshold, the model generates **28** false positives per **56,962** test transactions (a false positive rate of 0.049%).

Because the original features are PCA-anonymised, it is **not possible to conduct a standard fairness audit** (e.g., equalised false positive rates across demographic groups). Any deployment of this model in a consumer-facing context should:

1. Conduct a fairness audit on non-anonymised internal data before deployment
2. Implement a clear, fast dispute resolution process for flagged-but-legitimate customers
3. Monitor false positive rates by customer segment over time

---

## Ethical Considerations

- **Customer harm from FPs:** Over-aggressive fraud detection damages customer experience and trust
- **Customer harm from FNs:** Under-detection allows fraud losses that may not be fully reimbursed
- **Consent and transparency:** Customers in most jurisdictions have a right to know they are subject to automated decision-making (GDPR Article 22 in the EU)
- **Model governance:** This model should not be deployed without version control, monitoring, and a documented retraining schedule

---

## Caveats and Recommendations

- Retrain at minimum quarterly on recent data; monitor for feature distribution drift (e.g., PSI)
- Do not lower the threshold below the cost-optimal point without a corresponding increase in analyst review capacity
- Consider combining this model with the unsupervised autoencoder as a two-tier system: autoencoder as a broad anomaly net, supervised model for final scoring
