import os
import sys
import getpass  # Hides the password typing in the terminal for security
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
import joblib

print("🧠 Loading production AI assets...")
try:
    model = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("✅ System online.")
except FileNotFoundError:
    print("❌ Error: Missing model files in 'models/'. Please run train.py first.")
    sys.exit(1)

print("\n=========================================================================")
print("🤖 LIVE CREDIT DELINQUENCY AI AGENT - MULTI-OPERATOR WORKSPACE")
print("=========================================================================")

# 🔐 NEW: Dynamically request the Sender's email authentication details
print("Please authenticate your mailing host server parameters to begin:")
sender_email = input("✉️ Enter YOUR email address (to send from): ").strip()

# getpass.getpass makes sure that when the user types their password, characters do NOT show up on screen
sender_password = getpass.getpass("🔑 Enter YOUR 16-character App Password (hidden): ").strip()

# Automatically detect mail server based on the email domain provided
if "@gmail.com" in sender_email.lower():
    smtp_server = "smtp.gmail.com"
elif "@outlook.com" in sender_email.lower() or "@hotmail.com" in sender_email.lower():
    smtp_server = "smtp.office365.com"
else:
    print("\n💡 Custom email domain detected.")
    smtp_server = input("🌐 Enter your email provider's SMTP server (e.g., smtp.mail.yahoo.com): ").strip()

smtp_port = 587 # Standard secure TLS port
print("-------------------------------------------------------------------------")

# 2. Ask for the applicant file path
user_input_path = input("📥 Drag & drop or type your CSV path to begin audit: ").strip().strip("'\" ")

if not os.path.exists(user_input_path):
    print(f"❌ Error: Could not find file at '{user_input_path}'.")
    sys.exit(1)

df_user = pd.read_csv(user_input_path)

if 'Email' not in df_user.columns:
    print("❌ Validation Error: Uploaded dataset is missing 'Email' column.")
    sys.exit(1)

feature_names = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

df_features_only = df_user[feature_names]
features_scaled = scaler.transform(df_features_only)
probabilities = model.predict_proba(features_scaled)[:, 1]

bins = [-0.001, 0.299999, 0.699999, 1.0]
labels = ['Low Risk', 'Moderate Risk', 'High Risk']
risk_levels = pd.cut(probabilities, bins=bins, labels=labels)

df_final_report = df_user.copy()
df_final_report['Delinquency_Probability_%'] = np.round(probabilities * 100, 2)
df_final_report['Assessed_Risk_Level'] = risk_levels

print("\n--- 📈 BATCH PROCESSING COMPLETED ---")
print(df_final_report[['Email', 'Delinquency_Probability_%', 'Assessed_Risk_Level']].to_string(index=False))
print("-----------------------------------------")

print("\n🤖 Agent: 'Scanning database rows for Moderate Risk flag triggers...'")

for idx, row in df_final_report.iterrows():
    if row['Assessed_Risk_Level'] == 'Moderate Risk':
        customer_email = row['Email']
        prob_val = row['Delinquency_Probability_%']
        
        print(f"\n🚨 [TRIGGER ALERT]: Found Moderate Risk Profile -> {customer_email} ({prob_val}%)")
        permission = input(f"❓ Agent: 'Do you grant permission to send a LIVE email reminder to {customer_email}? (yes/no): ").strip().lower()
        
        if permission in ['yes', 'y']:
            print(f"🔄 Agent: 'Connecting to server as {sender_email} and routing pipeline packet...'")
            
            # Construct live email message structure
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = customer_email
            msg['Subject'] = "Urgent Account Notice: Financial Health Check-In"
            
            body = (
                f"Dear Customer,\n\n"
                f"This is an automated courtesy notice regarding your credit profile status.\n\n"
                f"Our real-time risk assessment indicators have noticed potential payment variances "
                f"on your recent monthly credit statements. To ensure your account parameters remain "
                f"within ideal operational thresholds, please review your outstanding balance statements.\n\n"
                f"Best Regards,\n"
                f"Credit Risk Automation Desk"
            )
            msg.attach(MIMEText(body, 'plain'))
            
            # Authenticate and broadcast email live over the web
            try:
                context = ssl.create_default_context()
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls(context=context) # Encrypt channel
                    server.login(sender_email, sender_password)
                    server.sendmail(sender_email, customer_email, msg.as_string())
                print(f"📧 Agent: 'SUCCESS! Live email sent from {sender_email} straight to {customer_email}.'")
            except Exception as e:
                print(f"❌ Agent Mail Error: Failed to transmit data payload. Details: {e}")
        else:
            print(f"⛔ Agent: 'Action overridden by user. Live email sequence aborted.'")

# Save output data file dynamically
input_filename = os.path.basename(user_input_path)
output_path = f"data/evaluated_{input_filename}"
df_final_report.to_csv(output_path, index=False)
print(f"\n💾 Core tasks complete! Evaluated file saved to: {output_path}\n")