import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("🔧 AI CNC Dashboard")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    data = pd.read_csv(file)

    st.write("Dataset Preview")
    st.write(data.head())

    X = data[['speed','feed','temperature','vibration','force']]
    y = data['tool_wear']

    model = LinearRegression()
    model.fit(X, y)

    data['Prediction'] = model.predict(X)

    st.line_chart(data[['tool_wear','Prediction']])

    st.write(data)
