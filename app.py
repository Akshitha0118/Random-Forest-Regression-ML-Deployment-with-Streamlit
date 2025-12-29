import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.markdown("<h1>🌲 Employee Salary Predictor (Random Forest)</h1>", unsafe_allow_html=True)

# ✅ CORRECT CSV LOADING
data = pd.read_csv("emp_sal.csv")

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

model = RandomForestRegressor(
    max_depth=4,
    criterion="poisson",
    random_state=0,
    n_estimators=6
)
model.fit(X, y)

level = st.slider("Select Experience Level", 1.0, 10.0, 6.5, 0.1)

if st.button("Predict Salary"):
    salary = model.predict([[level]])[0]
    st.success(f"💰 Predicted Salary: ₹ {salary:,.2f}")




