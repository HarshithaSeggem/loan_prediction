# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.title("Linear Regression App")

# Load dataset
try:
    data = pd.read_csv("task1_dataset.csv")
    st.success("Dataset loaded successfully ✅")
except:
    st.error("❌ Make sure task1_dataset.csv is in the same folder as app.py")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Select target
target = st.selectbox("Select Target Column", data.columns)

# ---------------- PREPROCESSING ---------------- #

# Separate target
y = data[target]
X = data.drop(columns=[target])

# ✅ Convert target if it's date/string
if y.dtype == 'object':
    try:
        y = pd.to_datetime(y, errors='coerce')
        y = y.map(pd.Timestamp.toordinal)
    except:
        pass

# ✅ Handle date column in features
if 'date' in X.columns:
    X['date'] = pd.to_datetime(X['date'], errors='coerce')
    X['date'] = X['date'].map(pd.Timestamp.toordinal)

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Handle missing values
X = X.fillna(X.mean())
y = pd.Series(y).fillna(y.mean())

# ---------------- MODEL ---------------- #

# Train-test split
test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

if st.button("Train Model"):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Load scaler (optional)
    try:
        scaler = pickle.load(open("scaler.pkl", "rb"))
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        st.success("Scaler applied ✅")
    except:
        scaler = None
        st.warning("No scaler found, using raw data")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("Model Performance")
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    # Save model
    st.session_state["model"] = model
    st.session_state["cols"] = X.columns
    st.session_state["scaler"] = scaler

# ---------------- PREDICTION ---------------- #

if "model" in st.session_state:

    st.subheader("Make Prediction")

    inputs = {}
    for col in st.session_state["cols"]:
        inputs[col] = st.number_input(f"Enter {col}", value=0.0)

    input_df = pd.DataFrame([inputs])

    # Ensure same column order
    input_df = input_df[st.session_state["cols"]]

    if st.button("Predict"):
        try:
            if st.session_state["scaler"]:
                input_df = st.session_state["scaler"].transform(input_df)

            pred = st.session_state["model"].predict(input_df)
            st.success(f"Prediction: {pred[0]}")
        except Exception as e:
            st.error(f"❌ Error: {e}")