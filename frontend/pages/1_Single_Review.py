import streamlit as st
import requests

st.title("Single Review Prediction")

review = st.text_area("Enter Review")

if st.button("Predict"):

    response = requests.post(

        "http://127.0.0.1:8000/predict",

        json={"review": review}

    )

    if response.status_code == 200:

        result = response.json()

        st.success("Prediction Completed")

        st.write(result)

    else:

        st.error("API Error")