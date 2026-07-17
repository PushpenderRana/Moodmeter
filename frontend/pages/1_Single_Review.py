import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")

st.title("Single Review Prediction")

review = st.text_area("Enter Review")

if st.button("Predict"):

    response = requests.post(
    f"{API_URL}/predict",
    json={"review": review}
    )

    if response.status_code == 200:

        result = response.json()

        st.success("Prediction Completed")

        st.write(result)

    else:

        st.error("API Error")