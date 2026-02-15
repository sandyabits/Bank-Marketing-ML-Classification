# 📌 Bank Marketing Prediction -- Machine Learning Classification

## 📖 Project Overview

This project focuses on predicting whether a customer will subscribe to
a term deposit based on marketing campaign data from a bank.

The problem is treated as a binary classification task, where the target
variable `deposit` indicates:

-   `yes` → Customer subscribed to term deposit\
-   `no` → Customer did not subscribe

The goal is to compare multiple machine learning models and deploy the
best-performing model using Streamlit Cloud.

------------------------------------------------------------------------

## 📂 Dataset Information

-   Dataset: Bank Marketing Dataset\
-   Source: UCI Machine Learning Repository\
-   Target Column: `deposit`\
-   Problem Type: Binary Classification

------------------------------------------------------------------------

## 🧹 Data Preprocessing

-   Handling categorical variables using encoding\
-   Feature scaling (where required)\
-   Train-test split (80:20)\
-   Model training on training data\
-   Evaluation on test data

------------------------------------------------------------------------

## 🤖 Machine Learning Models Implemented

1.  Logistic Regression\
2.  Decision Tree\
3.  K-Nearest Neighbors (KNN)\
4.  Naive Bayes\
5.  Random Forest\
6.  XGBoost

------------------------------------------------------------------------

## 📊 Model Evaluation Metrics

-   Accuracy\
-   Precision\
-   Recall\
-   F1 Score\
-   AUC-ROC Score\
-   MCC (Matthews Correlation Coefficient)\
-   Confusion Matrix

------------------------------------------------------------------------

## 🏆 Model Performance Comparison

  Model                 Accuracy   AUC     F1 Score   MCC
  --------------------- ---------- ------- ---------- -------
  Logistic Regression   0.695      0.758   0.636      0.392
  Decision Tree         0.641      0.639   0.617      0.279
  KNN                   0.679      0.716   0.637      0.356
  Naive Bayes           0.686      0.736   0.599      0.380
  Random Forest         0.719      0.775   0.675      0.438
  XGBoost               0.720      0.772   0.681      0.439

------------------------------------------------------------------------

## 🚀 Streamlit Deployment

Live Application:\
https://2025aa05827.streamlit.app/

The web application allows users to:

-   Upload the Bank Marketing dataset\
-   Select a machine learning model\
-   View predictions\
-   View performance metrics\
-   View confusion matrix

------------------------------------------------------------------------

## 📁 Project Structure

    Bank-Marketing-ML-Classification/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── models/
    │   ├── logistic.pkl
    │   ├── decision_tree.pkl
    │   ├── knn.pkl
    │   ├── naive_bayes.pkl
    │   ├── random_forest.pkl
    │   └── xgboost.pkl

------------------------------------------------------------------------

## ⚙️ How to Run Locally

1.  Clone Repository\
2.  Install Dependencies: pip install -r requirements.txt
3.  Run Streamlit App: streamlit run app.py

------------------------------------------------------------------------

## 📌 Technologies Used

-   Python\
-   Pandas\
-   NumPy\
-   Scikit-learn\
-   XGBoost\
-   Streamlit\
-   Joblib

------------------------------------------------------------------------

## 📚 Academic Conclusion

This project demonstrates:

-   End-to-end ML workflow\
-   Comparative model evaluation\
-   Model persistence\
-   Cloud deployment

Ensemble methods (Random Forest and XGBoost) showed superior performance
for this dataset.

------------------------------------------------------------------------

## 👤 Author

Sandya BK
