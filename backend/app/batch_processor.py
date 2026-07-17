import pandas as pd
from app.predictor import predict_sentiment

def process_csv(file):

    print("Processing CSV file...")

    df = pd.read_csv(file)

    print("CSV file read into DataFrame.")

    df["Review Text"] = df["Review Text"].fillna("")

    results = []

    for review in df["Review Text"]:

        if pd.isna(review):
            review = ""

        prediction = predict_sentiment(review)

        results.append(prediction)

    prediction_df = pd.DataFrame(results)

    output = pd.concat(
        [
            df.reset_index(drop=True),
            prediction_df.reset_index(drop=True)
        ],
        axis=1
    )

    return output