import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

model = joblib.load("models/final_model.pkl")

st.title("Loan Risk Decision Support System")

st.write("""
This system predicts the probability of credit card default using machine learning.

The prediction is based on financial attributes such as credit limit,
repayment history, bill amounts, and previous payments.

The tool is designed as a decision support system to help analyse
loan default risk.
""")

st.write("Enter client financial information below:")

LIMIT_BAL = st.number_input("Credit Limit", min_value=0)
SEX = st.selectbox("Sex (1 = Male, 2 = Female)", [1, 2])
EDUCATION = st.selectbox("Education Level", [1, 2, 3, 4])
MARRIAGE = st.selectbox("Marriage Status", [1, 2, 3])
AGE = st.number_input("Age", min_value=18)

PAY_0 = st.number_input("Repayment Status (Last Month)")
PAY_2 = st.number_input("Repayment Status (2 Months Ago)")
PAY_3 = st.number_input("Repayment Status (3 Months Ago)")
PAY_4 = st.number_input("Repayment Status (4 Months Ago)")
PAY_5 = st.number_input("Repayment Status (5 Months Ago)")
PAY_6 = st.number_input("Repayment Status (6 Months Ago)")

BILL_AMT1 = st.number_input("Bill Amount 1")
BILL_AMT2 = st.number_input("Bill Amount 2")
BILL_AMT3 = st.number_input("Bill Amount 3")
BILL_AMT4 = st.number_input("Bill Amount 4")
BILL_AMT5 = st.number_input("Bill Amount 5")
BILL_AMT6 = st.number_input("Bill Amount 6")

PAY_AMT1 = st.number_input("Payment Amount 1")
PAY_AMT2 = st.number_input("Payment Amount 2")
PAY_AMT3 = st.number_input("Payment Amount 3")
PAY_AMT4 = st.number_input("Payment Amount 4")
PAY_AMT5 = st.number_input("Payment Amount 5")
PAY_AMT6 = st.number_input("Payment Amount 6")

if st.button("Predict Loan Default Risk"):

    input_data = pd.DataFrame([[
        LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
        PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
        BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
        PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
    ]], columns=[
        'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11',
        'X12', 'X13', 'X14', 'X15', 'X16', 'X17',
        'X18', 'X19', 'X20', 'X21', 'X22', 'X23'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_percent = probability * 100

    st.write(f"Default Probability: {risk_percent:.2f}%")

    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        title={'text': "Default Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig)

    if prediction == 1:
        st.error("High Risk of Default")
    else:
        st.success("Low Risk of Default")