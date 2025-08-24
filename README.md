# 🏦 Loan Approval Classifier

This project applies **machine learning techniques** to predict whether a loan application should be **Approved** ✅ or **Rejected** ❌.  
It covers **data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, evaluation, and deployment** via a Streamlit app.

---

## 🔹 Problem Statement
Loan approval is traditionally a **manual and subjective** process.  
This project aims to build an **automated system** that improves decision speed, reduces bias, and enhances reliability.

---

## 📂 Dataset
The dataset contains:
- **Demographics**: Dependents, Education, Self-employed.  
- **Financials**: Annual income, Loan amount, Loan term, CIBIL score.  
- **Assets**: Residential, Commercial, Luxury, Bank.  
- **Target**: Loan status (Approved / Rejected).  

Files:
- `loan_approval_dataset.csv` → Full dataset  
- `train_data_loan_approval.csv` → Training set  
- `test_data_loan_approval.csv` → Testing set  
- `Sample (Try streamlit).csv` → Example input  

---

## 🧩 Methodology

### 🔹 Data Preprocessing
- Data cleaning & handling missing values  
- Outlier removal (IQR method)  
- Encoding categorical variables  
- Splitting into training & testing sets  

### 🔹 Exploratory Data Analysis (EDA)
- Visualization of categorical & numerical features  
- Relationship between features and loan status  
- Multivariate correlation analysis  

### 🔹 Modeling
- **Logistic Regression** (with and without SMOTE)  
- **Decision Tree Classifier** (baseline + tuned)  
- **Random Forest Classifier** (baseline + tuned)  
- **XGBoost Classifier**  

Hyperparameter tuning was performed using **GridSearchCV** for Decision Tree, Random Forest, and XGBoost.

---

## 📊 Results

| Model                      | Accuracy | AUC   | Notes                  |
|-----------------------------|----------|-------|------------------------|
| Logistic Regression         | 0.91     | 0.959 | Simple, interpretable  |
| Decision Tree               | 0.96     | 0.995 | Risk of overfitting    |
| Random Forest (Tuned)       | 0.98     | 0.999 | Best, robust & stable  |
| XGBoost (Tuned)             | 0.98     | 0.999 | Similar to RF, higher cost |

**Key Takeaways**:
- Random Forest and XGBoost delivered the **best results (~98% accuracy)**.  
- Logistic Regression is highly interpretable but weaker overall.  
- Decision Tree alone is prone to overfitting.  

---
## 💾 Model Saving & Deployment
- Models saved with **Joblib** as `.pkl` files:
  - `best_random_forest_model.pkl`  
  - `best_xgboost_model.pkl`  

- **Streamlit Application**:
  - Single prediction (manual applicant input).  
  - Batch predictions (CSV upload).  
  - Visualization of approval vs rejection distribution.  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loan-approval-project-by-nhe.streamlit.app/)

Run locally:
```bash
streamlit run App/App.py
```
---

By **N**ourelden **H**any **E**lhakiem


