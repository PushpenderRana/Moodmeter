import streamlit as st

st.title("📊 Dashboard")

if "prediction_df" not in st.session_state:

    st.warning("Please process a CSV first.")

else:

    df = st.session_state["prediction_df"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(df))

    col2.metric(
        "Positive",
        len(df[df["sentiment"] == "Positive"])
    )

    col3.metric(
        "Negative",
        len(df[df["sentiment"] == "Negative"])
    )

    col4.metric(
        "Urgent",
        len(df[df["urgent"] == True])
    )

    st.divider()

    st.subheader("Sentiment Distribution")
    st.bar_chart(df["sentiment"].value_counts())

    st.subheader("Complaint Categories")
    st.bar_chart(df["category"].value_counts())

    st.subheader("Assigned Teams")
    st.bar_chart(df["assigned_team"].value_counts())

    st.subheader("Priority Distribution")
    st.bar_chart(df["priority"].value_counts())

    st.subheader("Prediction Table")
    st.dataframe(df)