import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("📊 Bank Marketing Classification Models")
st.write("Predict whether a client will subscribe to a term deposit.")

# -----------------------------
# Model Selection
# -----------------------------

model_options = {
    "Logistic Regression": "models/Logistic_Regression.pkl",
    "Decision Tree": "models/Decision_Tree.pkl",
    "KNN": "models/KNN.pkl",
    "Naive Bayes": "models/Naive_Bayes.pkl",
    "Random Forest": "models/Random_Forest.pkl",
    "XGBoost": "models/XGBoost.pkl"
}

selected_model_name = st.selectbox("Select a Model", list(model_options.keys()))

# -----------------------------
# Upload Dataset
# -----------------------------

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=",")

    if "duration" in df.columns:
        df = df.drop("duration", axis=1)

    if "deposit" in df.columns:
        df["deposit"] = df["deposit"].str.lower().str.strip().map({"yes": 1, "no": 0})
        df_cleaned = df.dropna(subset=['deposit'])


        X = df_cleaned.drop("deposit", axis=1)
        y = df_cleaned["deposit"]

        # Load model
        model_path = model_options[selected_model_name]
        model = joblib.load(model_path)

        # Predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.3f}")
        col1.metric("AUC", f"{auc:.3f}")

        col2.metric("Precision", f"{precision:.3f}")
        col2.metric("Recall", f"{recall:.3f}")

        col3.metric("F1 Score", f"{f1:.3f}")
        col3.metric("MCC", f"{mcc:.3f}")

        # Confusion Matrix
        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        # Classification Report
        st.subheader("📄 Classification Report")
        report = classification_report(y, y_pred)
        st.text(report)

    else:
        st.error("Uploaded dataset must contain target column 'deposit'.")

else:
    st.info("Please upload a CSV file to proceed.")
