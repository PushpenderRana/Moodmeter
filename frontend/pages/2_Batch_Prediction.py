import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
st.title("📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    st.write("Preview")

    preview = pd.read_csv(uploaded_file)

    st.dataframe(preview.head())

    uploaded_file.seek(0)

    if st.button("Predict Reviews"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "text/csv"
            )
        }

        response = requests.post(
            f"{API_URL}/predict-batch",
            files=files)

        if response.status_code == 200:

            result_df = pd.read_csv(
            BytesIO(response.content)
            )

            # Save in Streamlit memory
            st.session_state["prediction_df"] = result_df

            st.success("Prediction Completed!")

            st.dataframe(result_df)

            st.download_button(
            "Download CSV",
            data=response.content,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        else:
            st.error(f"Prediction Failed ({response.status_code})")
            st.write(response.text)