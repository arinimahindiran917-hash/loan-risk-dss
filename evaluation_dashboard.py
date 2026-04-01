import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Loan Risk Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

.metric-title {
    font-size: 16px;
    color: #555;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #111;
}

</style>
""", unsafe_allow_html=True)


st.title("📊 Loan Risk Model Evaluation Dashboard")

st.write("""
Offline evaluation of machine learning models for credit default prediction.

This dashboard presents the evaluation results of the loan risk assessment system.
It includes model comparison, performance visualisations, feature importance,
and SHAP explainability for the selected final model.
""")

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "LightGBM",
        "CatBoost"
    ],
    "Accuracy": [0.8077, 0.8123, 0.8183, 0.8187, 0.8163],
    "Precision": [0.6868, 0.6310, 0.6634, 0.6671, 0.6539],
    "Recall": [0.2396, 0.3647, 0.3625, 0.3595, 0.3602],
    "F1 Score": [0.3553, 0.4623, 0.4688, 0.4672, 0.4645]
})

best_model = "Gradient Boosting"

st.subheader("Key Highlights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Selected Model</div>
        <div class="metric-value">{best_model}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Best Accuracy</div>
        <div class="metric-value">0.8187</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Best F1 Score</div>
        <div class="metric-value">0.4688</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Best Recall</div>
        <div class="metric-value">0.3647</div>
    </div>
    """, unsafe_allow_html=True)

st.subheader("Model Comparison Table")

st.dataframe(results.sort_values("F1 Score", ascending=False), use_container_width=True)

st.info("Gradient Boosting was selected as the final model because it achieved the highest F1-score, showing the best balance between precision and recall.")

st.subheader("Model Performance Comparison")

fig, ax = plt.subplots(figsize=(10,5))
results_plot = results.set_index("Model")[["Accuracy","Precision","Recall","F1 Score"]]
results_plot.plot(kind="bar", ax=ax)

ax.set_ylabel("Score")
ax.set_title("Comparison of Evaluation Metrics Across Models")

plt.xticks(rotation=20)

st.pyplot(fig)

roc_img = Image.open("models/roc_curve.png")
cm_img = Image.open("models/confusion_matrix.png")
shap_img = Image.open("models/shap_summary.png")

model = joblib.load("models/final_model.pkl")

feature_names = [
    "Credit Limit",
    "Gender",
    "Education",
    "Marital Status",
    "Age",
    "Repayment Status (Sep)",
    "Repayment Status (Aug)",
    "Repayment Status (Jul)",
    "Repayment Status (Jun)",
    "Repayment Status (May)",
    "Repayment Status (Apr)",
    "Bill Amount (Sep)",
    "Bill Amount (Aug)",
    "Bill Amount (Jul)",
    "Bill Amount (Jun)",
    "Bill Amount (May)",
    "Bill Amount (Apr)",
    "Payment Amount (Sep)",
    "Payment Amount (Aug)",
    "Payment Amount (Jul)",
    "Payment Amount (Jun)",
    "Payment Amount (May)",
    "Payment Amount (Apr)"
]

importance = model.feature_importances_

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False).head(10)

tab1, tab2, tab3 = st.tabs(["📈 Performance Visuals", "🧠 Explainability", "📝 Interpretation"])

with tab1:

    st.subheader("Performance Visualisations")

    col1, col2 = st.columns(2)

    with col1:
        st.image(roc_img, caption="ROC Curve", use_container_width=True)

    with col2:
        st.image(cm_img, caption="Confusion Matrix", use_container_width=True)

with tab2:

    st.subheader("Model Explainability")

    col3, col4 = st.columns(2)

    with col3:

        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1])
        ax2.set_title("Top 10 Most Important Features")
        ax2.set_xlabel("Importance Score")

        st.pyplot(fig2)

    with col4:
        st.image(shap_img, caption="SHAP Summary Plot", use_container_width=True)


with tab3:

    st.subheader("Interpretation of Results")

    st.markdown("""
### Main Findings

- **Gradient Boosting** achieved the strongest overall performance based on **F1-score**.
- **LightGBM** achieved the highest accuracy, but its F1-score was slightly lower.
- **Logistic Regression** showed the weakest recall, meaning it missed more actual defaulters.
- The **ROC curve** indicates that the model can distinguish between defaulters and non-defaulters reasonably well.
- The **confusion matrix** shows that the model correctly classified many non-default cases while identifying part of the default cases.
- The **feature importance chart** highlights the most influential financial variables used by the model.
- The **SHAP summary plot** provides interpretable evidence that repayment behaviour strongly influences default prediction.
""")

    st.success("These findings support the use of Gradient Boosting as the final model in the decision support system.")

st.markdown("---")
st.caption("MSc Project: Design and Evaluation of a Decision Support System for Loan Risk Assessment Using Secondary Financial Data")