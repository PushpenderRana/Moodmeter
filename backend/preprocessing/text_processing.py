import re
import string
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Stopwords
stop_words = set(stopwords.words("english"))

negations = {
    "not", "no", "nor",
    "don't", "didn't", "doesn't",
    "isn't", "aren't", "wasn't",
    "weren't", "won't", "can't",
    "couldn't", "shouldn't", "wouldn't"
}

stop_words -= negations

lemmatizer = WordNetLemmatizer()


def preprocess(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stop_words]

    tokens = [w for w in tokens if len(w) > 1]

    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)


if __name__ == "__main__":

    df = pd.read_csv("cleaned_reviews.csv")

    df["Cleaned_Review"] = df["Review Text"].apply(preprocess)

    print(df[["Review Text", "Cleaned_Review"]].head())

    df.to_csv("preprocessed_reviews.csv", index=False)

    print("✅ Preprocessing Completed")