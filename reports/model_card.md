# Model Card — Credit Card Fraud Detection

**Framework:** scikit-learn & XGBoost (`class_weight_XGB`, imbalanced-learn, SHAP)

---

## Model Details

| Property | Value |
|---|---|
| Final selected model | XGBoost (`class_weight_XGB`, scale_pos_weight=578.26) |
| Imbalance strategy | Cost-sensitive class weighting (`scale_pos_weight`) |
| Primary selection metric | Validation AUPRC |
| Dataset | ULB Credit Card Fraud Detection (Sep 2013) |
| Input features | V1–V28 (PCA-anonymised), Amount (scaled), Time (scaled) |
| Output | Fraud Risk Score ∈ [0, 1] |
| Operating decision threshold | 0.0600 (derived from cost optimization on Validation set) |

---

## Intended Use

**Intended users:** Fraud analysts, risk operations teams at financial institutions.

**Intended use cases:**
- Real-time scoring via FastAPI REST endpoint (`/predict`) or analytical review in Streamlit dashboard
- Portfolio demonstration of end-to-end statistically sound ML pipeline engineering
- Educational reference for imbalanced classification, leakage-free preprocessing, and cost-sensitive threshold selection

**Out-of-scope uses:**
- **Do not use** as an autonomous decision engine to execute irreversible account bans without human review
- **Do not use** on non-credit card domain data without retraining and domain validation
- **Do not use** as a certified fair model without auditing raw un-anonymised features

---

## Performance Metrics

### Validation Candidate Selection (60% Train / 20% Validation)
All candidate models were trained on `X_train` and evaluated on `X_val` at default threshold 0.50. `class_weight_XGB` achieved the highest Validation AUPRC (0.8153) and was selected.

### Final Untouched Test Set Evaluation (20% Test, 56,962 transactions, 98 fraud)

| Setting | Operating Threshold | Precision | Recall | F1 | Test AUPRC | Test ROC-AUC | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| Reference (Default) | 0.5000 | 0.9398 | 0.7959 | 0.8619 | 0.8771 | 0.9764 | 78 | 5 | 56,859 | 20 |
| **Validation-Locked** | **0.0600** | **0.8019** | **0.8673** | **0.8333** | **0.8771** | **0.9764** | **85** | **21** | **56,843** | **13** |

**Baseline comparison:** Logistic Regression baseline scored Validation AUPRC = 0.6831. The validation-selected XGBoost model improved Validation AUPRC by +0.1322 points and achieved Test AUPRC = 0.8771.

---

## Data Partitions & Leakage Guards

- **Dataset:** ULB Credit Card Fraud Detection (Kaggle: mlg-ulb/creditcardfraud)
- **Period:** September 2013, European cardholders
- **Size:** 284,807 transactions; 492 fraud (0.1727%)
- **Features:** PCA components V1–V28; raw Amount and Time
- **3-Way Stratified Split:** 
  - Train: 60.0% (170,883 transactions, 295 fraud)
  - Validation: 20.0% (56,962 transactions, 99 fraud)
  - Test: 20.0% (56,962 transactions, 98 fraud — Untouched until final scoring)
- **Preprocessing Isolation:** `StandardScaler` fitted strictly on `X_train`. Validation and test transformed using training mean ($87.63 Amount, 94,864.79s Time).
- **Cost Isolation:** FN cost ($130.15) computed strictly from `X_train` fraud rows.

---

## Limitations

1. **Temporal Horizon:** Trained on 2013 data. Performance on contemporary transactions subject to concept drift.
2. **Geographical Scope:** Reflects European cardholder patterns. Applicability to other regions unvalidated.
3. **Feature Anonymisation:** V1–V28 are redacted PCA components, preventing direct mapping to human-readable merchant categories.
4. **Cost Model Assumptions:** Operating threshold derived using $5 illustrative customer friction cost assumption.
5. **Random vs. Temporal Validation:** Uses stratified random split due to anonymised timestamps. Production evaluation should use time-based temporal split.

---

## Fairness & Governance

> [!WARNING]
> **False Positives cause customer friction.** At the locked 0.0600 threshold, the model generated **21** false positives out of **56,962** test transactions (a false positive rate of 0.0369%).

Because V1–V28 are anonymised PCA components, standard demographic fairness audits cannot be performed on this public dataset. Prior to production deployment, risk teams should:

1. Conduct fairness audits on un-anonymised internal attributes.
2. Maintain clear appeal and human review workflows for flagged legitimate transactions.
3. Monitor drift and re-evaluate thresholds based on observed operational capacity.
