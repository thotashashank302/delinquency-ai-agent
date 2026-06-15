import joblib
import pandas as pd
import numpy as np

# 1. Load the saved model and scaler from disk
print("🧠 Loading AI model and scaler assets...")
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("✅ Assets successfully loaded! System online.")

# 2. Define the exact feature names in the correct order
feature_names = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# 3. Simulate a brand new customer applying for credit
raw_data = [[
    50000,  # LIMIT_BAL: $50,000 credit limit
    2,      # SEX: Female
    2,      # EDUCATION: University
    1,      # MARRIAGE: Married
    35,     # AGE: 35 years old
    2,      # PAY_0: Delayed payments by 2 months last month (⚠️ HIGH RISK FLAG)
    2,      # PAY_2: Delayed payments by 2 months
    2,      # PAY_3: Delayed payments by 2 months
    2,      # PAY_4: Delayed payments by 2 months
    2,      # PAY_5: Delayed payments by 2 months
    2,      # PAY_6: Delayed payments by 2 months
    45000,  # BILL_AMT1: Maxed out bill amount near credit limit
    46000,  # BILL_AMT2
    47000,  # BILL_AMT3
    48000,  # BILL_AMT4
    49000,  # BILL_AMT5
    50000,  # BILL_AMT6
    0,      # PAY_AMT1: Paid $0 towards their debt last month
    0,      # PAY_AMT2: Paid $0
    1000,   # PAY_AMT3: Paid a tiny $1000 fraction
    0,      # PAY_AMT4: Paid $0
    0,      # PAY_AMT5: Paid $0
    0       # PAY_AMT6: Paid $0
]]

# Convert to a DataFrame with feature names to fix the warning!
new_customer = pd.DataFrame(raw_data, columns=feature_names)

# 4. Translate the new customer data using our saved scaler
new_customer_scaled = scaler.transform(new_customer)

# 5. Extract the exact probability of default
default_probability = model.predict_proba(new_customer_scaled)[0, 1]

# 6. Apply our strict banking custom risk threshold (0.35)
risk_threshold = 0.35
prediction = 1 if default_probability >= risk_threshold else 0

print("\n--- 📊 RISK ASSESSMENT REPORT ---")
print(f"👉 Default Probability Calculated: {default_probability * 100:.2f}%")
print(f"👉 Operational Risk Threshold:     {risk_threshold * 100:.2f}%")

if prediction == 1:
    print("❌ ALERT: Credit Application DENIED. High risk of delinquency.")
else:
    print("💚 SUCCESS: Credit Application APPROVED. Low risk account.")
print("---------------------------------")