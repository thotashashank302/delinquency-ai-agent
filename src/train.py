import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset cleanly
print("🔄 Loading clean dataset...")
df = pd.read_excel("data/raw/default of credit card clients.xls", header=1)

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=['ID', 'default payment next month'])
y = df['default payment next month']

# 3. Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the Features
print("⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and train the Random Forest Model
print("\n🌲 Training production Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
print("✅ Model Training Complete!")

# 6. THRESHOLD SHIFTING TO PUSH RECALL HIGHER
print("\n🔮 Applying custom risk threshold (0.35) to boost Recall...")
# Instead of predict(), we get raw probabilities for the default class (column 1)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# If probability is greater than 35%, flag as default (1), otherwise safe (0)
custom_threshold = 0.35
y_pred_custom = (probabilities >= custom_threshold).astype(int)

# 7. Print custom performance metrics
print(f"📊 New Accuracy Score: {accuracy_score(y_test, y_pred_custom) * 100:.2f}%")
print("\n📋 Upgraded Classification Report (Targeting High Recall):")
print(classification_report(y_test, y_pred_custom))

# 8. SAVE THE MODEL AND SCALER AS PHYSICAL FILES USING JOBLIB
print("\n💾 Saving model assets to disk using joblib...")
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("🎉 Success! Saved files safely to your Mac:")
print("   👉 models/random_forest_model.pkl (The Trained Model)")
print("   👉 models/scaler.pkl             (The Scaler Translator)")