import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Salary Predictor", layout="centered")

# ---------- MINIMAL CSS ----------
st.markdown("""
<style>
body {background-color:#f5f7fb;}
h1 {text-align:center; color:#1E88E5;}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>🌲 Employee Salary Predictor (Random Forest)</h1>", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
data = pd.read_csv(r'C:\Users\ADMIN\Downloads\23rd- Poly\23rd- Poly\1.POLYNOMIAL REGRESSION\emp_sal.csv')   # CSV must be in project folder
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# ---------- RANDOM FOREST MODEL ----------
model = RandomForestRegressor(
    max_depth=4,
    criterion="poisson",
    random_state=0,
    n_estimators=6
)
model.fit(X, y)

# ---------- USER INPUT ----------
level = st.slider("Select Experience Level", 1.0, 10.0, 6.5, 0.1)

# ---------- PREDICTION ----------
if st.button("Predict Salary"):
    salary = model.predict([[level]])[0]
    st.success(f"💰 Predicted Salary: ₹ {salary:,.2f}")

