from typing import Any, List

import joblib
import pandas as pd
import streamlit as st

from customer_churn_prediction.data import (
    FeaturePreprocessor,
    NUM_COLS,
    CAT_COLS,
)

MODEL_PATH = "models/stacking.joblib"


def predict(data: List[Any]) -> str:
    X = pd.DataFrame(
        [data],
        columns=NUM_COLS + CAT_COLS,
    )
    X = FeaturePreprocessor().transform(X)

    model = joblib.load(MODEL_PATH)

    prediction = "likely" if model.predict(X)[0] > 0.5 else "unlikely"

    return f"customer is {prediction} to churn"


if __name__ == "__main__":
    st.title("Customer Churn Prediction")
    data = [
        st.number_input("Client Period (months)", 0, 1200),
        st.number_input("Monthly Spending", 18.0, 120.0, step=0.01),
        st.number_input("Total Spent", 0.0, step=0.01),
        st.selectbox("Sex", ["Male", "Female"], index=0),
        st.selectbox("Senior Citizen", ["No", "Yes"], index=0),
        st.selectbox("Has Partner", ["No", "Yes"], index=0),
        st.selectbox("Has Child", ["No", "Yes"], index=0),
        st.selectbox("Has Phone Service", ["No", "Yes"], index=0),
        st.selectbox(
            "Has Multiple Phone Numbers",
            ["No", "Yes", "No phone service"],
            index=0,
        ),
        st.selectbox(
            "Has Internet Service", ["No", "DSL", "Fiber optic"], index=0
        ),
        st.selectbox(
            "Has Online Security Service",
            ["No", "Yes", "No internet service"],
            index=0,
        ),
        st.selectbox(
            "Has Online Backup", ["No", "Yes", "No internet service"], index=0
        ),
        st.selectbox(
            "Has Device Protection",
            ["No", "Yes", "No internet service"],
            index=0,
        ),
        st.selectbox(
            "Has Tech Support Access",
            ["No", "Yes", "No internet service"],
            index=0,
        ),
        st.selectbox(
            "Has Online TV", ["No", "Yes", "No internet service"], index=0
        ),
        st.selectbox(
            "Has Movie Subscription",
            ["No", "Yes", "No internet service"],
            index=0,
        ),
        st.selectbox(
            "Has Contract Phone",
            ["Month-to-month", "One year", "Two year"],
            index=0,
        ),
        st.selectbox("Has Paperless Billing", ["No", "Yes"], index=0),
        st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=0,
        ),
    ]

    if st.button("Predict"):
        st.success(predict(data))
