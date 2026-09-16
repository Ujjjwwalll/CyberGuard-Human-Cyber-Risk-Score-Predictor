# 🛡️ CyberGuard — Human Cyber-Risk Score Predictor

CyberGuard is an end-to-end Machine Learning capstone project that predicts an
employee's **cybersecurity risk level** (Low / Moderate / High / Severe) based on
their everyday digital security habits — password hygiene, MFA usage, phishing
awareness, device habits, and training engagement — and then lets them chat with a
**Gemini AI assistant** to understand and improve their score.

---

## 🎯 Problem Statement

Most cybersecurity breaches don't happen because of broken firewalls — they happen
because of **human behavior**: reused passwords, clicking suspicious links, skipping
MFA, oversharing on social media. Organizations run generic, one-size-fits-all
security training, but they rarely know **which employees are actually at higher risk**
before an incident happens.

CyberGuard solves this by turning a simple 17-question behavioral questionnaire into
a data-driven risk score, so organizations (or individuals) can identify risk **before**
it turns into an actual breach — and get personalized, explainable feedback instantly.

---

## ✨ Key Features

- **17-question behavioral assessment** covering passwords, phishing awareness, device
  habits, and security training engagement
- **Multi-class ML classifier** (Low / Moderate / High / Severe risk) trained and
  compared across 4 algorithms
- **Explainable results** — probability distribution chart across all 4 risk levels
- **Gemini AI integration** — a Gemini-powered cybersecurity assistant that explains
  *why* a user got their score and gives personalized tips
- **Interactive Streamlit UI** — clean form-based questionnaire with instant results
- **AI-powered assistance** — cybersecurity explanations and personalized guidance are
  generated through the Gemini API

---

## 🧰 Tech Stack

| Layer | Tools Used |
|---|---|
| Data handling & analysis | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Altair |
| Machine Learning | Scikit-learn, XGBoost |
| Model persistence | Joblib |
| Web App / UI | Streamlit |
| Generative AI | Google Gemini API, Gemini 3.6 Flash |
| Gemini SDK | google-genai |
| Statistical Testing | SciPy (Chi-Square test) |

---

## 📁 Project Structure

```
capstone-project/
├── cyberguard_raw_dataset.csv     # Raw survey dataset (10,000 rows)
├── CyberGuard_Notebook.ipynb      # Full ML pipeline notebook
├── app.py                          # Streamlit web application
├── risk_model.pkl                  # Trained & tuned XGBoost model
├── ordinal_encoder.pkl              # Fitted OrdinalEncoder
├── scaler.pkl                       # Fitted StandardScaler
├── ordinal_cols.pkl                 # List of ordinal column names
├── model_columns.pkl                # Exact column order expected by the model
├── risk_map.pkl                     # Label → number mapping
├── reverse_risk_map.pkl             # Number → label mapping
├── README.md                        # This file
```

---

## 📊 Dataset

- **10,000 raw survey responses**, cleaned down to **8,053 rows** after removing
  missing values and duplicates
- **17 input features**, all multiple-choice (no free text, no typos by design)
- **1 target column**: `risk_level` (Low / Moderate / High / Severe)
- Covers 4 behavioral themes: Passwords & Authentication, Email & Social
  Engineering, Devices & Network, and Awareness & Training

---

## 🤖 Model Comparison

Four classification models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **XGBoost (Tuned)** | **83.9%** | **83.9%** | **83.9%** | **83.8%** |
| XGBoost (Default) | 81.4% | 81.4% | 81.4% | 81.4% |
| Random Forest | 81.1% | 81.1% | 81.1% | 81.0% |
| Logistic Regression | 78.8% | 78.7% | 78.8% | 78.7% |
| K-Nearest Neighbors | 67.4% | 65.9% | 67.4% | 66.0% |

**Final model:** XGBoost, tuned via `GridSearchCV` (`n_estimators`, `max_depth`,
`learning_rate`), selected for its best performance across all four metrics.

---

## ⚙️ How It Works (Pipeline)

1. **Data Cleaning** — dropped missing values and duplicate submissions
2. **EDA** — univariate, bivariate, and multivariate analysis to understand
   behavioral patterns across departments and risk levels
3. **Encoding** — `OrdinalEncoder` for the 15 naturally-ordered questions
   (e.g. Never → Sometimes → Often → Always), one-hot encoding for `device_type`
   and `department`
4. **Feature Engineering** — a derived `security_awareness_score` combining
   several "good habit" columns
5. **Feature Selection** — Chi-Square test confirmed all features are
   statistically related to `risk_level`
6. **Train/Test Split** — 80/20 stratified split
7. **Scaling** — `StandardScaler` applied for consistency across all models
8. **Model Training & Comparison** — 4 models trained and benchmarked
9. **Hyperparameter Tuning** — `GridSearchCV` improved the best model further
10. **Deployment** — model + all preprocessing objects saved with `joblib`,
    then wired into a Streamlit UI with a Gemini AI chat assistant

---

## 🚀 Running the Project

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit matplotlib seaborn altair google-genai
```

### 2. Set up the Gemini API

Create a Gemini API key through Google AI Studio and store it securely.

For **Streamlit Cloud**, add the following to your app's Secrets:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do **not** hard-code your API key directly inside `app.py`.

The application uses the `google-genai` Python SDK to communicate with Gemini. Google currently recommends the Google GenAI SDK for Gemini API applications.

The project uses **Gemini 3.6 Flash** as the AI assistant model.

### 3. (Optional) Re-run the notebook
Open `CyberGuard_Notebook.ipynb` in Jupyter to see the full data science
pipeline from raw CSV to trained model.

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 🔮 Future Improvements

- Add SHAP-based explainability for individual predictions (feature-level "why")
- Multi-user support with historical score tracking over time
- Admin dashboard for organization-wide risk aggregation by department
- Support for additional Gemini models via a model-selection dropdown

---

## 👤 Author

Built as an academic capstone ML project — covering the complete lifecycle from
raw data to a deployed, AI-assisted, interactive application.
