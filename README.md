# 💳 Credit Card Fraud Detection using Machine Learning & SHAP Explainability

## 🧠 Overview
This project aims to build an **AI-powered Credit Card Fraud Detection System** using machine learning models and interpretability tools like **SHAP (SHapley Additive exPlanations)**.  
It identifies potentially fraudulent transactions from highly imbalanced financial data, improving transparency and decision trust.

---

## 📂 Dataset
**Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- Total samples: **284,807 transactions**
- Features: **30 anonymized PCA features (V1–V28)** + `Amount`, `Time`
- Target variable: **Class**  
  - `0` → Legitimate  
  - `1` → Fraudulent  
- Fraud percentage: ~0.17% (highly imbalanced dataset)

---

## ⚙️ Data Preprocessing
1. Removed duplicates → `(283,726 × 31)`
2. Standardized `Amount` and `Time`
3. Used **Stratified Train-Test Split**
4. Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance minority class

---

## 🤖 Models Used
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------|-----------|------------|---------|-----------|----------|
| Logistic Regression | 0.9995 | 0.972 | 0.737 | 0.838 | 0.9758 |
| Random Forest | 0.9995 | 0.913 | 0.768 | 0.834 | 0.9694 |
| XGBoost | 0.9991 | 0.728 | 0.789 | 0.758 | 0.9700 |
| **XGBoost (Tuned)** | **0.9995** | **0.972** | **0.737** | **0.838** | **0.9758** |

✅ **XGBoost (Tuned)** gave the best overall performance with the highest ROC-AUC score of **0.9758**.

---

## 📈 Results Visualization
The following visualizations were generated in the notebooks:
- **Confusion Matrix** for each model
- **ROC Curves** comparing model performance
- **Feature Importance Plots** for Random Forest and XGBoost
- **SHAP Analysis Visuals:**
  - Global Feature Importance Bar Plot  
  - SHAP Summary Plot  
  - Waterfall Plot (local explanation)  
  - Comparison of Correctly Detected vs Missed Frauds  

---

## 🕵️ Analysis and Interpretation of SHAP Results for Credit Card Fraud Detection

The provided SHAP analysis interprets how individual features influence the credit card fraud detection model's predictions.

---

### 1. Global Feature Importance (Average Impact)
- **Top Features:** `V14`, `V4`, `V12`, and `V8`
- **V14’s Dominance:** The single most important predictor, with the highest SHAP magnitude.

### 2. Feature Impact Distribution (Summary Plot)
- **V14:** High (red) values → Fraud; Low (blue) → Legitimate  
- **V4, V12, V8:** Also strong indicators with opposite directional influence.

### 3. Local Explanation (Waterfall Plot)
- **Case 845 Prediction:** True label FRAUD, predicted FRAUD with 0.9355 probability.  
- **Top contributors:** `V14 (-4.004)`, `V10 (-3.295)`, `V12 (-4.024)` strongly pushed prediction toward *fraud*.

### 4. Missed Frauds Analysis
- **V14, V10, V4, V12** show high SHAP impact in correctly detected frauds, but near-zero for missed frauds.  
- Missed frauds mimic legitimate transactions → low SHAP contribution → misclassification.

### 5. Detailed Case Comparison
| Case | Prediction | Fraud Prob | Key SHAP Impact |
|------|-------------|-------------|-----------------|
| #1 (Detected) | FRAUD | 0.9959 | Strong positive SHAPs (`V14, V10, V12, V16`) |
| #4 & #5 (Missed) | LEGITIMATE | 0.0001 / 0.0000 | Weak, negative SHAPs (`V14, V12, V10, V4`) |

### 📝 Key Takeaway
- **Critical Features:** `V14`, `V4`, `V12`, `V10`
- **Reason for Misses:** Fraudulent transactions with subdued feature patterns appear legitimate.

---

## 🔍 Evaluation Summary
- The model achieves **near-perfect accuracy** but is evaluated on **precision–recall trade-off** due to data imbalance.  
- The tuned XGBoost offers the best ROC-AUC balance and interpretability through SHAP.

---

## 🧩 Technologies Used
- **Python 3**
- **Libraries:** NumPy, Pandas, Scikit-Learn, XGBoost, Imbalanced-Learn, Matplotlib, Seaborn, SHAP
- **Environment:** Google Colab / Jupyter Notebook

---
Future Work
Integrate real-time fraud detection pipeline with streaming data

Experiment with deep learning (Autoencoders, LSTM)

Deploy REST API for model inference

Further enhance interpretability using LIME and counterfactuals
