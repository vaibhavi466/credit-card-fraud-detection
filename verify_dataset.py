import pandas as pd

df = pd.read_csv('data/raw/creditcard.csv')
n_rows = len(df)
n_fraud = int(df['Class'].sum())
n_legit = n_rows - n_fraud
fraud_pct = n_fraud / n_rows * 100

row_match = "MATCH OK" if n_rows == 284807 else "MISMATCH - STOP"
fraud_match = "MATCH OK" if n_fraud == 492 else "MISMATCH - STOP"

print(f"Rows:        {n_rows:,}   expected=284,807  [{row_match}]")
print(f"Fraud cases: {n_fraud}        expected=492       [{fraud_match}]")
print(f"Legit cases: {n_legit:,}")
print(f"Fraud rate:  {fraud_pct:.4f}%")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print(f"Shape: {df.shape}")

mean_fraud_amount = df.loc[df['Class'] == 1, 'Amount'].mean()
mean_legit_amount = df.loc[df['Class'] == 0, 'Amount'].mean()
print(f"Mean fraud Amount: ${mean_fraud_amount:.2f}")
print(f"Mean legit Amount: ${mean_legit_amount:.2f}")

if n_rows != 284807 or n_fraud != 492:
    print("\nERROR: Dataset does not match expected ULB creditcard specs. Stopping.")
    import sys
    sys.exit(1)
else:
    print("\nDataset verified successfully. Proceeding with pipeline.")
