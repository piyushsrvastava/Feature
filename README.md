# Feature
Email Validation: Check email validity using MailboxLayer API (SMTP, format, disposable checks).
Company Enrichment: Fetch company details from Crunchbase + B2B data APIs (industry, location, employees, funding).
AI Lead Scoring: Predict lead conversion likelihood using a pre-trained Random Forest classifier.
Export: Download enriched and scored leads as CSV.

Multi-threaded & Efficient: Validates multiple emails in parallel.

Safe Feature Engineering: Handles missing or inconsistent data gracefully.
#Data set & Model Training
1. Data set Requirement
   CSV file named dataset.csv

  Columns:
    company, email, linkedin_url, funding_stage, industry, location, employees,       converted (target)
2. Training Script Overview:
  # Feature engineering
- Binary columns: linkedin_present, professional_email
- Numerical: employees
- Categorical: funding_stage, industry, location
- Dropped: company, email, linkedin_url

## Preprocessing pipeline
- Impute missing categorical with most frequent
- One-hot encode categorical
- Median impute numerical + StandardScaler
- Pass-through binary features

## Model
- RandomForestClassifier(n_estimators=300, class_weight="balanced")
3. Train & Save Model
  from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

## Evaluate
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

## Save model
joblib.dump(model, "model.pkl")


# install depandancy 
streamlit==1.30.0
pandas==2.1.0
joblib==1.3.2
requests==2.32.0
altair==5.0.1
scikit-learn==1.3.2

# Usage 
1. Run the Streamlit App:
   streamlit run app.py
2.Open your browser at http://localhost:8501.

3.Upload a CSV file containing leads. Minimum columns expected:

company

email

Optional: domain, linkedin_url

4.Navigate through the tabs:

 Validate: Validate all emails

 Enrich: Add company/industry info

 Score: Display AI lead score and adjusted score

 Export: Download enriched & scored leads

 #Configuration
 MODEL_PATH: Path to your ML model (model.pkl)

MAILBOXLAYER_API_KEY: API key for email validation

RAPID_KEY: API key for Crunchbase & B2B data enrichment

# How It's work 
1. Email Validation: Uses MailboxLayer API to check SMTP, format, disposable status.

2.Company Enrichment:

Crunchbase → get company permalink → fetch details

B2B API for domain-based enrichment

3. Lead Scoring:

Preprocess features (categorical, numerical, binary)

Predict probability using RandomForestClassifier

Adjust score based on email validity

4. Export: Merge enriched info and scores, then allow CSV download
 
